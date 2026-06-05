use std::error::Error;
use std::{sync::Arc, time::Duration};
use std::default::Default;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;
use vulkano::{buffer::BufferContents, command_buffer::allocator::StandardCommandBufferAllocator, image::view::ImageView, pipeline::{ DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo, graphics::{GraphicsPipelineCreateInfo, color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::RasterizationState, vertex_input::{Vertex, VertexDefinition}, viewport::{Viewport, ViewportState}}, layout::PipelineDescriptorSetLayoutCreateInfo}, render_pass::{self, Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}};
use vulkano_util::{context::{VulkanoConfig, VulkanoContext}, window::VulkanoWindows};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop};



struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    rcx: Option<RenderContext>
}


struct RenderContext {
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
}


fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());

        let windows = VulkanoWindows::default();

        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        let vertices = [
            MyVertex {
                position: [-0.5, -0.25],
            },
            MyVertex {
                position: [0.0, 0.5],
            },
            MyVertex {
                position: [0.25, -0.1],
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        App { context, windows, command_buffer_allocator, vertex_buffer, rcx: None, }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        self.windows
            .create_window(event_loop, &self.context, &Default::default(), |_| {});
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let window_size = window_renderer.window().inner_size();

        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec2 position;

                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) out vec4 f_color;

                    void main() {
                        f_color = vec4(1.0, 0.0, 0.0, 1.0);
                    }
                ",
            }
        }

        let render_pass = vulkano::single_pass_renderpass!(
            self.context.device().clone(),
            attachments: {
                color: {
                    format: window_renderer.swapchain_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        ).unwrap();

        let framebuffers = window_size_dependent_setup(window_renderer.swapchain_image_views(), &render_pass);

        let pipeline = {
            let vs = vs::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.context.device().clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.context.device().clone())
                    .unwrap(),
            ).unwrap();

            let subass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                self.context.device().clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                }
            ).unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        self.rcx = Some(RenderContext {
            render_pass,
            framebuffers,
            pipeline,
            viewport
        });

    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    )
    {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                window_renderer.resize();
            }
            WindowEvent::RedrawRequested => {
                let window_size = window_renderer.window().inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                let previous_frame_end = window_renderer.acquire(Some(Duration::from_millis(1000)), |swapchain_images| {
                    rcx.framebuffers = window_size_dependent_setup(swapchain_images, &rcx.render_pass);
                    rcx.viewport.extent = window_size.into();
                }).unwrap();

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit
                ).unwrap();


                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[window_renderer.image_index() as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    ).unwrap()
                    .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap();

                unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

                builder
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(self.context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                window_renderer.present(future, false);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.windows.get_primary_renderer_mut().unwrap().window().request_redraw();
    }
}


#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn window_size_dependent_setup(
    swapchain_images: &[Arc<ImageView>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_images
        .iter()
        .map(|swapchain_image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![swapchain_image.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
