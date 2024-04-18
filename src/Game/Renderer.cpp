//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Renderer.hpp"
// Strawberry Window
#include "Strawberry/Window/Window.hpp"
// Strawberry Core
#include "Strawberry/Core/Math/Transformations.hpp"


//======================================================================================================================
//  Shader Code
//======================================================================================================================
static uint8_t vertexShader[] =
{
	#include "Board.vert.bin"
};


static uint8_t fragmentShader[] =
{
	#include "Board.frag.bin"
};


//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	Renderer::Renderer(const Window::Window& window)
		: mRenderSize(window.GetSize().AsType<float>())
		, mInstance()
		, mPhysicalDevice(&mInstance.GetPhysicalDevices()[0])
		, mQueueFamily(mPhysicalDevice->SearchQueueFamilies(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)[0])
		, mDevice(*mPhysicalDevice, {Vulkan::QueueCreateInfo(mQueueFamily, 1)})
		, mAllocator(mDevice)
		, mQueue(mDevice.GetQueue(mQueueFamily, 0))
		, mSurface(window, mDevice)
		, mSwapchain(*mQueue, mSurface, window.GetSize(), RLPUYO_REALTIME ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_IMMEDIATE_KHR)
		, mRenderPass(Vulkan::RenderPass::Builder(mDevice)
							  .WithColorAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR,
												   VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
												   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
												   Strawberry::Core::Math::Vec4f(),
												   VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_NONE_EXT)
			.WithSubpass(Vulkan::SubpassDescription().WithColorAttachment(0))
			.Build())
		, mFramebuffer(mRenderPass, &mAllocator, window.GetSize().AsType<unsigned int>())
		, mPipelineLayout(CreatePipelineLayout(mDevice, mRenderPass))
		, mPipeline(CreatePipeline(mRenderSize.AsType<float>(), mDevice, mRenderPass, mPipelineLayout))
		, mDescriptorPool(mDevice, 0, 4, {VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1), VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)})
		, mConstantsDescriptor(mDescriptorPool, mPipelineLayout.GetSetLayout(0))
		, mConstantsBuffer(&mAllocator, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 16 * sizeof(float) + 2 * sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
		, mBoardStateDescriptor {
			Vulkan::DescriptorSet(mDescriptorPool, mPipelineLayout.GetSetLayout(1)),
			Vulkan::DescriptorSet(mDescriptorPool, mPipelineLayout.GetSetLayout(1))}
		, mBoardStateBuffer {
			Vulkan::Buffer(&mAllocator, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 2 * sizeof(uint32_t) + 2 * sizeof(int32_t) + (BOARD_WIDTH * BOARD_HEIGHT + 2) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
			Vulkan::Buffer(&mAllocator, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 2 * sizeof(uint32_t) + 2 * sizeof(int32_t) + (BOARD_WIDTH * BOARD_HEIGHT + 2) * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)}
		, mCommandPool(*mQueue)
		, mPrimaryCommandBuffer(mCommandPool)
		, mDrawingBuffer(mCommandPool, VK_COMMAND_BUFFER_LEVEL_SECONDARY)
	{
		mConstantsDescriptor.SetUniformBuffer(0, 0, mConstantsBuffer);
		mBoardStateDescriptor[0].SetStorageBuffer(0, 0, mBoardStateBuffer[0]);
		mBoardStateDescriptor[1].SetStorageBuffer(0, 0, mBoardStateBuffer[1]);

		auto projection = Core::Math::Orthographic<float>(0.0, mRenderSize[0], mRenderSize[1], 0.0, -1.0, 1.0);
		auto boardSize = Core::Math::Vector<uint32_t, 2>(BOARD_WIDTH, BOARD_HEIGHT);

		Core::IO::DynamicByteBuffer constantsBufferContents;
		constantsBufferContents.Push(projection.Transposed());
		constantsBufferContents.Push(boardSize);
		mConstantsBuffer.SetData(constantsBufferContents);
	}


	Renderer::~Renderer()
	{
		if (mQueue)
		{
			mQueue->WaitUntilIdle();
		}
	}


	void Renderer::Submit(uint32_t boardIndex, const Board& board)
	{
		Core::IO::DynamicByteBuffer mBoardStateBufferContents = Core::IO::DynamicByteBuffer::WithCapacity(mBoardStateBuffer[boardIndex].GetSize());
		mBoardStateBufferContents.Push(boardIndex);
		mBoardStateBufferContents.Push<uint32_t>(0);
		mBoardStateBufferContents.Push(board.FallingTilesPosition().UnwrapOr(TilePosition(0, 0)));
		for (int y = BOARD_HEIGHT - 1; y >= 0; --y)
		{
			for (int x = 0; x < BOARD_WIDTH; x++)
			{
				mBoardStateBufferContents.Push(static_cast<uint32_t>(board.GetTile({x, y})));
			}
		}
		mBoardStateBufferContents.Push(static_cast<uint32_t>(board.FallingTilesBottom().UnwrapOr(Tile::EMPTY)));
		mBoardStateBufferContents.Push(static_cast<uint32_t>(board.FallingTilesTop().UnwrapOr(Tile::EMPTY)));
		mBoardStateBuffer[boardIndex].SetData(mBoardStateBufferContents);



		mQueue->WaitUntilIdle();

		if (mDrawingBuffer.State() != Vulkan::CommandBufferState::Recording)
		{
			mDrawingBuffer.Begin(true, mRenderPass, 0);
			mDrawingBuffer.BindPipeline(mPipeline);
			mDrawingBuffer.BindDescriptorSet(mPipeline, 0, mConstantsDescriptor);
		}

		mDrawingBuffer.BindDescriptorSet(mPipeline, 1, mBoardStateDescriptor[boardIndex]);
		mDrawingBuffer.Draw(6, BOARD_WIDTH * BOARD_HEIGHT + 2);
	}


	void Renderer::Render()
	{
		mQueue->WaitUntilIdle();
		mPrimaryCommandBuffer.Reset();

		auto* swapchainImage = mSwapchain.GetNextImage().Unwrap();
		mPrimaryCommandBuffer.Begin(true);
		mPrimaryCommandBuffer.BeginRenderPass(mRenderPass, mFramebuffer, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

		if (mDrawingBuffer.State() == Vulkan::CommandBufferState::Recording)
		{
			mDrawingBuffer.End();
			mPrimaryCommandBuffer.ExcecuteSecondaryBuffer(mDrawingBuffer);
		}

		mPrimaryCommandBuffer.EndRenderPass();
		mPrimaryCommandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_UNDEFINED)
				.ToLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		});
		mPrimaryCommandBuffer.BlitImage(mFramebuffer.GetColorAttachment(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR);
		mPrimaryCommandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
				.ToLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
		});
		mPrimaryCommandBuffer.End();
		mQueue->Submit(mPrimaryCommandBuffer);
		mSwapchain.Present();
	}


	Vulkan::PipelineLayout
	Renderer::CreatePipelineLayout(const Vulkan::Device& device, const Vulkan::RenderPass& renderPass)
	{
		return Vulkan::PipelineLayout::Builder(device)
			.WithDescriptorSet({VkDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr)})
			.WithDescriptorSet({VkDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr)})
			.Build();
	}


	Vulkan::GraphicsPipeline Renderer::CreatePipeline(Core::Math::Vec2f renderSize, const Vulkan::Device& device,
													  const Vulkan::RenderPass& renderPass,
													  const Vulkan::PipelineLayout& layout)
	{
		return Vulkan::GraphicsPipeline::Builder(layout, renderPass, 0)
			.WithShaderStage(VK_SHADER_STAGE_VERTEX_BIT,
							 Vulkan::Shader::Compile(device, {vertexShader, sizeof(vertexShader)}).Unwrap())
			.WithShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT,
							 Vulkan::Shader::Compile(device, {fragmentShader, sizeof(fragmentShader)}).Unwrap())
			.WithInputAssembly(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
			.WithRasterization(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE)
			.WithViewport({VkViewport(0.0, renderSize[1], renderSize[0], -renderSize[1], 0.0, 1.0)}, {VkRect2D({0, 0}, {static_cast<uint32_t>(renderSize[0]), static_cast<uint32_t>(renderSize[1])})})
			.WithColorBlending({VkPipelineColorBlendAttachmentState{
				.blendEnable = VK_TRUE,
				.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
				.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
				.colorBlendOp = VK_BLEND_OP_ADD,
				.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
				.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
				.alphaBlendOp = VK_BLEND_OP_ADD,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
			}})
			.Build();
	}
}
