use std::default::Default;
use std::error::Error;
use std::mem;
use std::sync::Arc;

use vulkano::Version;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::ImageAspects;
use vulkano::image::ImageCreateFlags;
use vulkano::image::ImageLayout;
use vulkano::image::ImageSubresourceRange;
use vulkano::image::sampler::Filter;
use vulkano::image::sampler::SamplerAddressMode::Repeat;
use vulkano::image::sampler::SamplerCreateInfo;
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::instance::InstanceExtensions;
use vulkano::instance::debug::DebugUtilsMessageSeverity;
use vulkano::instance::debug::DebugUtilsMessageType;
use vulkano::instance::debug::DebugUtilsMessenger;
use vulkano::instance::debug::DebugUtilsMessengerCallback;
use vulkano::instance::debug::DebugUtilsMessengerCreateInfo;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use vulkano::{VulkanError, VulkanLibrary};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano_taskgraph::descriptor_set::BindlessContext;
use vulkano_taskgraph::descriptor_set::SamplerId;
use vulkano_taskgraph::graph::{
    AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph,
};
use vulkano_taskgraph::resource::ResourcesCreateInfo;
use vulkano_taskgraph::resource::{self, AccessTypes, Flight, Resources};
use vulkano_taskgraph::{Id, resource_map};
use winit::event_loop::ControlFlow;
use winit::window::Window;
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop};

use crate::graphics::TriangleTask;

mod compute;
mod graphics;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    rcx: Option<RenderContext>,
    _debug_callback: Option<DebugUtilsMessenger>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    viewport: Viewport,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,
    virtual_swapchain_id: Id<Swapchain>,
    src_image_id: Id<Image>,
    dst_image_id: Id<Image>,
    src_sampler_id: SamplerId,
    dst_sampler_id: SamplerId,
    src_storage_image_id: vulkano_taskgraph::descriptor_set::StorageImageId,
    dst_storage_image_id: vulkano_taskgraph::descriptor_set::StorageImageId,
    src_sampled_image_id: vulkano_taskgraph::descriptor_set::SampledImageId,
    dst_sampled_image_id: vulkano_taskgraph::descriptor_set::SampledImageId,
    virtual_src_image_id: Id<Image>,
    virtual_dst_image_id: Id<Image>,
}

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new() }.unwrap();

        let required_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..Surface::required_extensions(event_loop)
        };
        let layers = ["VK_LAYER_KHRONOS_validation"];

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_layers: &layers,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let _debug_callback = unsafe {
            DebugUtilsMessenger::new(
                &instance,
                &DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::ERROR
                        | DebugUtilsMessageSeverity::WARNING
                        | DebugUtilsMessageSeverity::INFO
                        | DebugUtilsMessageSeverity::VERBOSE,
                    message_type: DebugUtilsMessageType::GENERAL
                        | DebugUtilsMessageType::VALIDATION
                        | DebugUtilsMessageType::PERFORMANCE,
                    ..DebugUtilsMessengerCreateInfo::new(&DebugUtilsMessengerCallback::new(
                        |message_severity, message_type, callback_data| {
                            let severity = if message_severity
                                .intersects(DebugUtilsMessageSeverity::ERROR)
                            {
                                "error"
                            } else if message_severity
                                .intersects(DebugUtilsMessageSeverity::WARNING)
                            {
                                "warning"
                            } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
                                "information"
                            } else if message_severity
                                .intersects(DebugUtilsMessageSeverity::VERBOSE)
                            {
                                "verbose"
                            } else {
                                panic!("no-impl");
                            };

                            let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
                                "general"
                            } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
                                "validation"
                            } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
                                "performance"
                            } else {
                                panic!("no-impl");
                            };

                            println!(
                                "{} {} {}: {}",
                                callback_data.message_id_name.unwrap_or("unknown"),
                                ty,
                                severity,
                                callback_data.message
                            );
                        },
                    ))
                },
            )
        }
        .ok();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
            ..BindlessContext::required_extensions(&instance)
        };

        let device_features = BindlessContext::required_features(&instance);

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_1 || p.supported_extensions().khr_maintenance2
            })
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
                    && p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // todo: devices do not always have a graphics/compute with present.
                        // this should be two checks, one for compute and one for graphics.
                        // +transfer?
                        q.queue_flags
                            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER )
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_1 {
            device_extensions.khr_maintenance2 = true;
        }

        if physical_device.api_version() < Version::V1_2
            && physical_device.supported_extensions().khr_image_format_list
        {
            device_extensions.khr_image_format_list = true;
        }

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_extensions: &device_extensions,
                enabled_features: &device_features,
                // todo: get compute queue also +transfer?
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let resources = Resources::new(
            &device,
            &ResourcesCreateInfo {
                bindless_context: Some(&Default::default()),
                ..Default::default()
            },
        )
        .unwrap();

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            rcx: None,
            _debug_callback,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let bcx = self.resources.bindless_context().unwrap();

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        let swapchain_format;
        let swapchain_id = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            (swapchain_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            self.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        min_image_count: surface_capabilities
                            .min_image_count
                            .max(MIN_SWAPCHAIN_IMAGES),
                        image_format: swapchain_format,
                        image_extent: window_size.into(),
                        image_usage: ImageUsage::COLOR_ATTACHMENT,
                        composite_alpha: surface_capabilities
                            .supported_composite_alpha
                            .into_iter()
                            .next()
                            .unwrap(),

                        ..Default::default()
                    },
                )
                .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_src_image_id = task_graph.add_image(&ImageCreateInfo {
            image_type: vulkano::image::ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            // todo: maybe remove storage from src_img
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,// | ImageUsage::STORAGE,
            ..Default::default()
        });

        let virtual_dst_image_id = task_graph.add_image(&ImageCreateInfo {
            image_type: vulkano::image::ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: /* ImageUsage::TRANSFER_DST | */ ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        });

        let src_image_id = self
            .resources
            .create_image(
                &ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    extent: [1024, 1024, 1],
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::STORAGE,
                    // todo: look at init code for image
                    // initial_layout: ImageLayout::Preinitialized,
                    ..Default::default()
                },
                &Default::default(),
            )
            .unwrap();

        let dst_image_id = self
            .resources
            .create_image(
                &ImageCreateInfo {
                    flags: ImageCreateFlags::MUTABLE_FORMAT,
                    image_type: vulkano::image::ImageType::Dim2d,
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    extent: [1024, 1024, 1],
                    usage: /* ImageUsage::TRANSFER_DST | */ ImageUsage::SAMPLED | ImageUsage::STORAGE,
                    ..Default::default()
                },
                &Default::default(),
            )
            .unwrap();

        let src_sampler_id = bcx
            .global_set()
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
                address_mode: [Repeat, Repeat, Repeat],
                ..Default::default()
            })
            .unwrap();

        let dst_sampler_id = bcx
            .global_set()
            .create_sampler(&SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
                address_mode: [Repeat, Repeat, Repeat],
                ..Default::default()
            })
            .unwrap();

        let src_storage_image_id = bcx
            .global_set()
            .create_storage_image(
                src_image_id,
                &ImageViewCreateInfo {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        ..Default::default()
                    },
                    usage: ImageUsage::STORAGE,
                    ..Default::default()
                },
                ImageLayout::General,
            )
            .unwrap();

        let dst_storage_image_id = bcx
            .global_set()
            .create_storage_image(
                dst_image_id,
                &ImageViewCreateInfo {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        ..Default::default()
                    },
                    usage: ImageUsage::STORAGE,
                    ..Default::default()
                },
                ImageLayout::General,
            )
            .unwrap();

        let src_sampled_image_id = bcx
            .global_set()
            .create_sampled_image(
                src_image_id,
                &ImageViewCreateInfo {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        ..Default::default()
                    },
                    usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
                    ..Default::default()
                },
                ImageLayout::General,
            )
            .unwrap();

        let dst_sampled_image_id = bcx
            .global_set()
            .create_sampled_image(
                dst_image_id,
                &ImageViewCreateInfo {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        ..Default::default()
                    },
                    usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
                    ..Default::default()
                },
                ImageLayout::General,
            )
            .unwrap();

        let compute_node_id = task_graph
            .create_task_node(
                "Compute",
                vulkano_taskgraph::QueueFamilyType::Compute,
                compute::ComputeTask::new(&self, src_image_id),
            )
            .image_access(
                virtual_dst_image_id,
                AccessTypes::COMPUTE_SHADER_STORAGE_WRITE,
                resource::ImageLayoutType::General,
            )
            .image_access(
                virtual_src_image_id,
                AccessTypes::COMPUTE_SHADER_SAMPLED_READ,
                resource::ImageLayoutType::General,
            )
            .build();

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });

        let virtual_frame_buffer_id = task_graph.add_framebuffer();

        let triangle_node_id = task_graph
            .create_task_node(
                "Triangle",
                vulkano_taskgraph::QueueFamilyType::Graphics,
                graphics::TriangleTask::new(self, virtual_swapchain_id),
            )
            .framebuffer(virtual_frame_buffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_WRITE,
                resource::ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .image_access(
                virtual_dst_image_id,
                AccessTypes::FRAGMENT_SHADER_SAMPLED_READ,
                resource::ImageLayoutType::General,
            )
            .build();

        task_graph
            .add_edge(compute_node_id, triangle_node_id)
            .unwrap();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        let triangle_node = task_graph.task_node_mut(triangle_node_id).unwrap();
        let subpass = triangle_node.subpass().unwrap().clone();
        triangle_node
            .task_mut()
            .downcast_mut::<TriangleTask>()
            .unwrap()
            .create_pipeline(self, &subpass);

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            viewport,
            recreate_swapchain,
            task_graph,
            virtual_swapchain_id,
            virtual_src_image_id,
            virtual_dst_image_id,
            src_image_id,
            dst_image_id,
            src_sampler_id,
            dst_sampler_id,
            src_storage_image_id,
            dst_storage_image_id,
            src_sampled_image_id,
            dst_sampled_image_id,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                if rcx.recreate_swapchain {
                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..*create_info
                        })
                        .expect("failed to recreate swapchain");

                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                let flight = self.resources.flight(self.flight_id);
                flight.wait(None).unwrap();

                let resource_map = resource_map!(&rcx.task_graph,
                    rcx.virtual_swapchain_id => rcx.swapchain_id,
                    rcx.virtual_src_image_id => rcx.src_image_id,
                    rcx.virtual_dst_image_id => rcx.dst_image_id,
                )
                .unwrap();

                match unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
                } {
                    Ok(()) => {
                        mem::swap(&mut rcx.src_image_id, &mut rcx.dst_image_id);
                        mem::swap(&mut rcx.src_sampler_id, &mut rcx.dst_sampler_id);
                        mem::swap(&mut rcx.src_storage_image_id, &mut rcx.dst_storage_image_id);
                        mem::swap(&mut rcx.src_sampled_image_id, &mut rcx.dst_sampled_image_id);
                    }
                    Err(ExecuteError::Swapchain {
                        error: VulkanError::OutOfDate,
                        ..
                    }) => {
                        rcx.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}
