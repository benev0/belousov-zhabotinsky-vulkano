use std::{slice, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
    },
    render_pass::Subpass,
    swapchain::Swapchain,
};
use vulkano_taskgraph::{ClearValues, Id, Task, TaskContext, resource::HostAccessType};

use crate::{App, RenderContext};

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub(crate) struct TriangleTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer_id: Id<Buffer>,
    swapchain_id: Id<Swapchain>,
}

impl TriangleTask {
    pub fn new(app: &mut App, swapchain_id: Id<Swapchain>) -> Self {
        let vertices = [
            MyVertex {
                position: [-1.0, -1.0],
            },
            MyVertex {
                position: [-1.0, 1.0],
            },
            MyVertex {
                position: [1.0, 1.0],
            },
            MyVertex {
                position: [1.0, -1.0],
            },
        ];

        let vertex_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_, tcx| {
                    tcx.try_write_buffer::<[MyVertex]>(vertex_buffer_id, ..)?
                        .copy_from_slice(&vertices);
                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                // todo: access types below?
                [],
                [],
            )
        }
        .unwrap();

        let pipeline = None;

        Self {
            pipeline,
            vertex_buffer_id,
            swapchain_id,
        }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        let pipeline = {
            let vs = unsafe { vs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(&app.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            let bcx = app.resources.bindless_context().unwrap();

            let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

            GraphicsPipeline::new(
                &app.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState {
                        topology: vulkano::pipeline::graphics::input_assembly::PrimitiveTopology::TriangleFan,
                        ..Default::default()
                    }),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        self.pipeline = Some(pipeline);
    }
}

impl Task for TriangleTask {
    type World = RenderContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        clear_values.set(
            self.swapchain_id.current_image_id(),
            [2.0 / 255.0, 6.0 / 255.0, 24.0 / 255.0, 1.0],
        );
    }

    unsafe fn execute(
        &self,
        cbf: &mut vulkano_taskgraph::command_buffer::RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        unsafe { cbf.set_viewport(0, slice::from_ref(&rcx.viewport)) };

        unsafe { cbf.bind_pipeline_graphics(self.pipeline.as_ref().unwrap()) };
        unsafe { cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]) };

        unsafe {
            cbf.push_constants(
                self.pipeline.as_ref().unwrap().layout(),
                0,
                &fs::PushConstantData {
                    sampler_id: rcx.dst_sampler_id,
                    dst_image: rcx.dst_sampled_image_id,
                },
            )
        };

        unsafe { cbf.draw(4, 1, 0, 0) };

        Ok(())
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shader.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shader.frag",
    }
}
