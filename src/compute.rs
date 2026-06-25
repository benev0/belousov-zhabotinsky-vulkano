use std::{slice, sync::Arc};

use rand::{RngExt, rngs::ThreadRng};
use vulkano::{
    buffer::{BufferContents, BufferCreateInfo, BufferUsage},
    image::{Image, ImageAspects, ImageSubresourceLayers, ImageSubresourceRange},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, Pipeline, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
    },
};
use vulkano_taskgraph::{
    Id, Task, TaskContext,
    command_buffer::{BufferImageCopy, CopyBufferToImageInfo, DependencyInfo, ImageMemoryBarrier},
    resource::HostAccessType,
};

use crate::{App, RenderContext};

#[derive(Debug, Clone, Copy, BufferContents)]
#[repr(C)]
struct Pixel {
    position: [u8; 4],
}

impl Pixel {
    fn new_random(rng: &mut ThreadRng) -> Self {
        Self {
            position: [rng.random(), rng.random(), rng.random(), 255],
        }
    }
}

pub(crate) struct ComputeTask {
    pipeline: Arc<ComputePipeline>,
}

impl ComputeTask {
    pub fn new(app: &App, image: Id<Image>) -> Self {
        let bcx = app.resources.bindless_context().unwrap();

        let mut rng = rand::rng();

        let pixels: Vec<Pixel> = (0..1024 * 1024)
            .into_iter()
            .map(|_| Pixel::new_random(&mut rng))
            .collect();

        let upload_buffer = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(pixels.as_slice()).unwrap(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |rcb, tcx| {
                    tcx.try_write_buffer::<[Pixel]>(upload_buffer, ..)?
                        .copy_from_slice(&pixels);

                    rcb.pipeline_barrier(&DependencyInfo {
                        image_memory_barriers: &[ImageMemoryBarrier {
                            old_layout: vulkano::image::ImageLayout::Undefined,
                            new_layout: vulkano::image::ImageLayout::General,
                            image: image,
                            subresource_range: ImageSubresourceRange {
                                aspects: ImageAspects::COLOR,
                                ..Default::default()
                            },
                            ..Default::default()
                        }],
                        ..Default::default()
                    });

                    rcb.copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: upload_buffer,
                        dst_image: image,
                        dst_image_layout: vulkano_taskgraph::resource::ImageLayoutType::General,
                        regions: &[BufferImageCopy {
                            image_extent: [1024, 1024, 1],
                            image_subresource: ImageSubresourceLayers {
                                aspects: ImageAspects::COLOR,
                                ..Default::default()
                            },
                            ..Default::default()
                        }],
                        ..Default::default()
                    });

                    Ok(())
                },
                [(upload_buffer, HostAccessType::Write)],
                // todo: access types below?
                [],
                [],
            )
        }
        .unwrap();

        let pipeline = {
            let cs = unsafe { cs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(&cs);

            let layout = bcx
                .pipeline_layout_from_stages(slice::from_ref(&stage))
                .unwrap();

            ComputePipeline::new(
                &app.device,
                None,
                &ComputePipelineCreateInfo::new(stage, &layout),
            )
            .unwrap()
        };

        Self { pipeline }
    }
}

impl Task for ComputeTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        unsafe { cbf.bind_pipeline_compute(&self.pipeline) };

        unsafe {
            cbf.push_constants(
                self.pipeline.layout(),
                0,
                &cs::PushConstantData {
                    sampler_id: world.src_sampler_id,
                    src_image: world.src_sampled_image_id,
                    dst_image: world.dst_storage_image_id,
                },
            )
        };

        unsafe { cbf.dispatch([128, 128, 1]) };

        Ok(())
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shader.comp",
    }
}
