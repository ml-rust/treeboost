//! GPU device initialization and buffer management for WGPU backend.
//!
//! Handles async device creation, buffer allocation, and command submission.

use std::sync::Arc;
use std::time::Duration;
use wgpu::{
    Adapter, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePipeline, Device,
    DeviceDescriptor, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions,
};

/// GPU device wrapper with device, queue, and adapter info.
pub struct GpuDevice {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub adapter: Adapter,
    pub limits: Limits,
    /// Whether subgroup operations are supported
    pub subgroups_supported: bool,
    /// Minimum subgroup size (0 if not supported)
    pub min_subgroup_size: u32,
    /// Maximum subgroup size (0 if not supported)
    pub max_subgroup_size: u32,
}

impl GpuDevice {
    /// Attempt to create a GPU device. Returns None if no suitable GPU is found.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        // Create WGPU instance with all backends
        let instance = Instance::default();

        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        // Adapter info (also exposes subgroup size range in wgpu 29+)
        let info = adapter.get_info();

        // Get device limits
        let limits = adapter.limits();

        // Check for subgroup support
        let adapter_features = adapter.features();
        let subgroups_supported = adapter_features.contains(wgpu::Features::SUBGROUP);

        // Build required features (include SUBGROUP if available)
        let required_features = if subgroups_supported {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };

        // Request device with compute features (wgpu 27 API)
        let (device, queue) = match adapter
            .request_device(&DeviceDescriptor {
                label: Some("TreeBoost GPU Device"),
                required_features,
                required_limits: Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
        {
            Ok(result) => result,
            Err(_) => return None,
        };

        // Get subgroup size range from adapter info (moved off Limits in wgpu 29)
        let (min_subgroup_size, max_subgroup_size) = if subgroups_supported {
            (info.subgroup_min_size, info.subgroup_max_size)
        } else {
            (0, 0)
        };

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
            limits,
            subgroups_supported,
            min_subgroup_size,
            max_subgroup_size,
        })
    }

    /// Get the device name for logging.
    pub fn name(&self) -> String {
        self.adapter.get_info().name
    }

    /// Get the backend type (Vulkan, Metal, DX12, etc.).
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter.get_info().backend
    }

    /// Create a storage buffer for GPU data.
    pub fn create_storage_buffer(&self, label: &str, size: u64, read_write: bool) -> Buffer {
        let usage = if read_write {
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
        } else {
            BufferUsages::STORAGE | BufferUsages::COPY_DST
        };

        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer for shader parameters.
    pub fn create_uniform_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for CPU readback.
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Upload data to a buffer.
    pub fn write_buffer<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &[T]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Create a command encoder.
    pub fn create_encoder(&self, label: &str) -> CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Submit commands and wait for completion.
    pub fn submit_and_wait(&self, encoder: CommandEncoder) {
        let submission = self.queue.submit(std::iter::once(encoder.finish()));
        // wgpu 27: PollType::Wait with submission_index
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        });
    }

    /// Submit commands asynchronously (non-blocking).
    ///
    /// Returns a submission index that can be used with `wait_for_submission`.
    /// Use this for double-buffering: submit work for the next batch while
    /// the current batch is still processing.
    pub fn submit_async(&self, encoder: CommandEncoder) -> wgpu::SubmissionIndex {
        self.queue.submit(std::iter::once(encoder.finish()))
    }

    /// Wait for a specific submission to complete.
    pub fn wait_for_submission(&self, submission: wgpu::SubmissionIndex) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        });
    }

    /// Poll the device without blocking (for checking completion).
    pub fn poll(&self) -> bool {
        self.device
            .poll(wgpu::PollType::Poll)
            .map(|status| status.is_queue_empty())
            .unwrap_or(false)
    }

    /// Read buffer data back to CPU (synchronous/blocking).
    pub fn read_buffer<T: bytemuck::Pod>(&self, staging: &Buffer, output: &mut [T]) {
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });

        {
            let data = slice.get_mapped_range();
            let src: &[T] = bytemuck::cast_slice(&data);
            output.copy_from_slice(&src[..output.len()]);
        }

        staging.unmap();
    }

    /// Read partial buffer data back to CPU.
    ///
    /// Reads only the first `output.len()` elements from the staging buffer.
    /// Useful when the actual data size is smaller than the buffer capacity.
    pub fn read_buffer_partial<T: bytemuck::Pod>(&self, staging: &Buffer, output: &mut [T]) {
        let byte_size = std::mem::size_of_val(output) as u64;
        let slice = staging.slice(..byte_size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });

        {
            let data = slice.get_mapped_range();
            let src: &[T] = bytemuck::cast_slice(&data);
            output.copy_from_slice(src);
        }

        staging.unmap();
    }

    /// Create a compute pipeline from WGSL source.
    pub fn create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> ComputePipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None, // Auto-derive from shader
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Try to create a compute pipeline, returning None if shader compilation fails.
    /// This is used for optional features like subgroups that may not be supported.
    pub fn try_create_compute_pipeline(
        &self,
        label: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> Option<ComputePipeline> {
        // Use catch_unwind to handle shader compilation errors gracefully
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.create_compute_pipeline(label, shader_source, entry_point)
        }))
        .ok()
    }

    /// Get maximum workgroup size (typically 256 or 1024).
    pub fn max_workgroup_size(&self) -> u32 {
        self.limits.max_compute_workgroup_size_x
    }

    /// Get maximum storage buffer binding size.
    pub fn max_storage_buffer_size(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        // This test will be skipped on systems without GPU
        if let Some(device) = GpuDevice::new() {
            println!(
                "GPU device created: {} ({:?})",
                device.name(),
                device.backend()
            );
            assert!(device.max_workgroup_size() >= 256);
        } else {
            println!("No GPU available, skipping test");
        }
    }
}
