#include <stdio.h>
#include <assert.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

typedef uint8_t u8;
typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;

#define SHADERS_FOLDER "shaders/"
const auto VK_COLOR_COMPONENT_RGBA_BITS = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

struct Vert {
	glm::vec2 pos;
	glm::vec3 color;
};
struct Ubo {
	glm::vec3 color;
};
const Vert triangle[] = {
	{{-0.8, +0.8}, {1, 0, 0}},
	{{+0.8, +0.8}, {0, 1, 0}},
	{{   0, -0.9}, {0, 0, 1}},
};

struct Vk {
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkPhysicalDeviceProperties physicalDeviceProps;
	VkPhysicalDeviceMemoryProperties deviceMemProps;
	VkDevice device;
	VkSurfaceKHR surface;
	u32 graphicsQueueFamily;
	VkQueue graphicsQueue;
	VkCommandPool graphicsCmdPool;
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	VkImageView swapchainImgViews[2];
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	VkFramebuffer framebuffers[2];
	VkCommandBuffer cmdBuffers[2];
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkSemaphore semaphore_swapchainImgAvailable[2];
	VkSemaphore semaphore_drawFinished[2];
	VkFence fence_queueWorkFinished[2];
	VkDescriptorSet descriptorSets[2];
	VkBuffer ubos[2];
	VkDeviceMemory ubosMemory;
};
static Vk vk;
static GLFWwindow* window;

struct Buffer {
	u32 len;
	u8* data;
};
static Buffer loadBinaryFile(const char* fileName)
{
	FILE* file = fopen(fileName, "rb");
	if (!file)
		return {};
	fseek(file, 0, SEEK_END);
	const u32 len = ftell(file);
	fseek(file, 0, SEEK_SET);
	u8* data = new u8[len];
	fread(data, 1, len, file);
	fclose(file);
	return { len, data };
}

static VkShaderModule loadSpirvShaderModule(const char* fileName)
{
	Buffer buffer = loadBinaryFile(fileName);
	assert(buffer.data);

	VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = buffer.len,
		.pCode = (u32*)buffer.data
	};
	VkShaderModule shaderModule;
	vkCreateShaderModule(vk.device, &info, nullptr, &shaderModule);

	delete[] buffer.data;
	return shaderModule;
}

static VkPipelineShaderStageCreateInfo makeShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule module)
{
	return VkPipelineShaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = stage,
		.module = module,
		.pName = "main",
		.pSpecializationInfo = nullptr // allows to specify values for shader constants
	};
}

static void createSwapchainAndFramebuffers(u32 screenW, u32 screenH)
{
	VkResult vkRes;
	auto oldSwapchain = vk.swapchain;
	{ // create swapchain
		VkSwapchainCreateInfoKHR info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = vk.surface,
			.minImageCount = 2,
			.imageFormat = VK_FORMAT_B8G8R8A8_SRGB,
			.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
			.imageExtent = {screenW, screenH},
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, // only once queue will reference an image at the same time. We can still reference the image from different queues, but not at the same time. We must use a memory barrier for this!
			.queueFamilyIndexCount = 1,
			.pQueueFamilyIndices = &vk.graphicsQueueFamily,
			.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_MAILBOX_KHR, // with 2 images, MAILBOX and FIFO are equivalent
			.clipped = VK_TRUE, // this allows to discard the rendering of hidden pixel regions. E.g a window is partially covered by another
			.oldSwapchain = oldSwapchain,
		};
		vkRes = vkCreateSwapchainKHR(vk.device, &info, nullptr, &vk.swapchain);
		assert(vkRes == VK_SUCCESS);
	}

	// destroy oldSwapchain stuff
	if (oldSwapchain != VK_NULL_HANDLE) {
		for (int i = 0; i < 2; i++) {
			vkDestroySemaphore(vk.device, vk.semaphore_swapchainImgAvailable[i], nullptr);
			vkDestroySemaphore(vk.device, vk.semaphore_drawFinished[i], nullptr);
			// 
			vkDestroyFramebuffer(vk.device, vk.framebuffers[i], nullptr);
			vkDestroyImageView(vk.device, vk.swapchainImgViews[i], nullptr);
		}
		vkDestroySwapchainKHR(vk.device, oldSwapchain, nullptr);
	}

	{ // create image views of the swapchain
		VkImage images[2];
		u32 numImages = 2;
		vkRes = vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &numImages, images);
		assert(vkRes == VK_SUCCESS);
		assert(numImages == 2);

		for (u32 i = 0; i < 2; i++) {
			VkImageViewCreateInfo info = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = images[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_SRGB,
				//.components = VK_COMPONENT_SWIZZLE_IDENTITY,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				}
			};
			vkRes = vkCreateImageView(vk.device, &info, nullptr, &vk.swapchainImgViews[i]);
			assert(vkRes == VK_SUCCESS);
		}
	}

	// -- create framebuffers
	for (int i = 0; i < 2; i++)
	{
		const VkFramebufferCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = vk.renderPass, // compatible renderPass
			.attachmentCount = 1,
			.pAttachments = vk.swapchainImgViews + i,
			.width = screenW,
			.height = screenH,
			.layers = 1,
		};
		vkRes = vkCreateFramebuffer(vk.device, &info, nullptr, &vk.framebuffers[i]);
		assert(vkRes == VK_SUCCESS);
	}
}

static void createSemaphores()
{
	VkResult vkRes;
	// -- create swapchain semaphores
	const VkSemaphoreCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
	};
	for (int i = 0; i < 2; i++) {
		vkRes = vkCreateSemaphore(vk.device, &info, nullptr, &vk.semaphore_swapchainImgAvailable[i]);
		assert(vkRes == VK_SUCCESS);
		vkRes = vkCreateSemaphore(vk.device, &info, nullptr, &vk.semaphore_drawFinished[i]);
		assert(vkRes == VK_SUCCESS);
	}
}

static void allocateCmdBuffers()
{
	const VkCommandBufferAllocateInfo info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk.graphicsCmdPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 2,
	};
	const VkResult vkRes = vkAllocateCommandBuffers(vk.device, &info, vk.cmdBuffers);
	assert(vkRes == VK_SUCCESS);
}

static void recordCmdBuffers(u32 screenW, u32 screenH)
{
	VkResult vkRes;
	for (int i = 0; i < 2; i++) {
		const VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		vkRes = vkBeginCommandBuffer(vk.cmdBuffers[i], &beginInfo);
		assert(vkRes == VK_SUCCESS);

		const VkClearValue CLEAR_VALUE = { .color = {.float32 = {0.2f, 0.2f, 0.2f, 0.f}} };
		const VkRenderPassBeginInfo renderPassBeginInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = vk.renderPass,
			.framebuffer = vk.framebuffers[i],
			.renderArea = { {0, 0}, {screenW, screenH} },
			.clearValueCount = 1, // one clear value for each attachment in the framebuffer
			.pClearValues = &CLEAR_VALUE
		};
		vkCmdBeginRenderPass(vk.cmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(vk.cmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);

		const VkViewport viewport = { 0.f, 0.f, screenW, screenH, 0.f, 1.f };
		vkCmdSetViewport(vk.cmdBuffers[i], 0, 1, &viewport);
		const VkRect2D scissorRegion = { {0, 0}, {screenW, screenH} };
		vkCmdSetScissor(vk.cmdBuffers[i], 0, 1, &scissorRegion);

		// bind (descriptor sets) uniforms
		vkCmdBindDescriptorSets(vk.cmdBuffers[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout,
			0, 1, vk.descriptorSets,
			0, nullptr // dynamic offsets
		);
		// draw the triangle!
		u64 vertBufferOffset = 0;
		vkCmdBindVertexBuffers(vk.cmdBuffers[i], 0, 1, &vk.vertexBuffer, &vertBufferOffset);
		vkCmdDraw(vk.cmdBuffers[i],
			3, // num vertices
			1, // num instances
			0, // first vertex
			0 // first instance
		);

		vkCmdEndRenderPass(vk.cmdBuffers[i]);

		vkRes = vkEndCommandBuffer(vk.cmdBuffers[i]);
		assert(vkRes == VK_SUCCESS);
	}
}

void example_1()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(800, 600, "example 1", nullptr, nullptr);

	VkResult vkRes;
	{ // -- create instance
		u32 numRequiredExtensions;
		const char** requiredExtensions = glfwGetRequiredInstanceExtensions(&numRequiredExtensions);
		const VkApplicationInfo appInfo = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0 };
		const VkInstanceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = numRequiredExtensions,
			.ppEnabledExtensionNames = requiredExtensions
		};
		vkRes = vkCreateInstance(&info, nullptr, &vk.instance);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create device
		VkPhysicalDevice physicalDevices[8];
		u32 numPhysicalDevices = std::size(physicalDevices);
		vkRes = vkEnumeratePhysicalDevices(vk.instance, &numPhysicalDevices, physicalDevices);
		assert(vkRes == VK_SUCCESS);
		assert(numPhysicalDevices);
		
		auto calcPropsScore = [](const VkPhysicalDeviceProperties& props) -> int {
			switch (props.deviceType) {
				case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 0;
				case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
				case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
				case VK_PHYSICAL_DEVICE_TYPE_CPU: return 3;
				case VK_PHYSICAL_DEVICE_TYPE_OTHER: default: return 4;
			}
		};
		auto calcMem = [](const VkPhysicalDeviceMemoryProperties& props) {
			u64 mem = 0;
			for (u32 i = 0; i < props.memoryHeapCount; i++) {
				if (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
					mem = glm::max(mem, props.memoryHeaps[i].size);
			}
			return mem;
		};

		// select best physical device
		u32 bestPhysicalDevice = 0;
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[0], &props);
		VkPhysicalDeviceMemoryProperties memProps;
		vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &memProps);
		int bestPropsScore = calcPropsScore(props);
		u64 bestMem = calcMem(memProps);
		for (u32 i = 1; i < numPhysicalDevices; i++) {
			vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
			const int score = calcPropsScore(props);
			if (score < bestPropsScore) {
				bestPhysicalDevice = i;
				bestPropsScore = score;
			}
			else if (score == bestPropsScore) {
				vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProps);
				const u64 mem = calcMem(memProps);
				if (mem > bestMem) {
					bestPhysicalDevice = i;
					bestMem = mem;
				}
			}
		}
		vk.physicalDevice = physicalDevices[bestPhysicalDevice];
		vkGetPhysicalDeviceProperties(vk.physicalDevice, &vk.physicalDeviceProps);
		vkGetPhysicalDeviceMemoryProperties(vk.physicalDevice, &vk.deviceMemProps);

		glfwCreateWindowSurface(vk.instance, window, nullptr, &vk.surface);

		VkQueueFamilyProperties familyProps[8];
		u32 numQueueFamilies = std::size(familyProps);
		vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &numQueueFamilies, familyProps);
		vk.graphicsQueueFamily = numQueueFamilies;
		for (u32 i = 0; i < numQueueFamilies; i++) {
			const bool supportsGraphics = familyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
			VkBool32 supportsSurface;
			vkGetPhysicalDeviceSurfaceSupportKHR(vk.physicalDevice, i, vk.surface, &supportsSurface);
			// in real life HW there is no driver that provides graphics and presentation as separate families: https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
			if (supportsGraphics && supportsSurface) {
				vk.graphicsQueueFamily = i;
				break;
			}
		}
		assert(vk.graphicsQueueFamily != numQueueFamilies);

		const char* deviceExtensionNames[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
		const float queuePriorities[] = { 1.f };
		VkDeviceQueueCreateInfo queueInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = vk.graphicsQueueFamily,
			.queueCount = 1,
			.pQueuePriorities = queuePriorities,
		};
		const VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queueInfo,
			.enabledExtensionCount = std::size(deviceExtensionNames),
			.ppEnabledExtensionNames = deviceExtensionNames
			// enabledLayerCount, ppEnabledLayerNames, pEnabledFeatures
		};
		vkRes = vkCreateDevice(vk.physicalDevice, &info, nullptr, &vk.device);
		assert(vkRes == VK_SUCCESS);

		vkGetDeviceQueue(vk.device, vk.graphicsQueueFamily, 0, &vk.graphicsQueue);
	}

	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &surfaceCapabilities);
	const auto [screenW, screenH] = surfaceCapabilities.currentExtent;

	{ // -- create renderpass
		const VkAttachmentDescription attachment = {
			.format = VK_FORMAT_B8G8R8A8_SRGB,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, // this informs the driver what layout to expect at the beginning of the renderPass
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR // this tells the driver to perform a layout transition at the end of the renderPass
		};
		const VkAttachmentReference inputAttachmentRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }; // this provoques a layout transition
		const VkSubpassDescription subpass {
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &inputAttachmentRef,
			// .pResolveAttachments = ,
			// .pDepthStencilAttachment = ,
			// .preserveAttachmentCount = , // when we have multiple subpasses we need to explicitly tell the driver that the contents of unused attachments must be preserved
			// .pPreserveAttachments = ,
		};
		VkRenderPassCreateInfo info = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = 0,
			.pDependencies = nullptr
		};
		vkRes = vkCreateRenderPass(vk.device, &info, nullptr, &vk.renderPass);
		assert(vkRes == VK_SUCCESS);

		{ // -- create graphics pipeline
			VkShaderModule vertShad = loadSpirvShaderModule(SHADERS_FOLDER"example_1_vert.spirv");
			VkShaderModule fragShad = loadSpirvShaderModule(SHADERS_FOLDER"simple_frag.spirv");

			const VkPipelineShaderStageCreateInfo shaderStages[2] = {
				makeShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShad),
				makeShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShad),
			};

			const VkVertexInputBindingDescription vertInputBinding = {
				.binding = 0,
				.stride = sizeof(Vert),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			};
			const VkVertexInputAttributeDescription vertInputAttribs[] = {
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = offsetof(Vert, pos),
				},
				{
					.location = 1,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = offsetof(Vert, color),
				},
			};
			const VkPipelineVertexInputStateCreateInfo vertexInputInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &vertInputBinding,
				.vertexAttributeDescriptionCount = std::size(vertInputAttribs),
				.pVertexAttributeDescriptions = vertInputAttribs
			};

			const VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				.primitiveRestartEnable = VK_FALSE
			};

			const VkViewport viewport = {
				.x = 0.f,
				.y = 0.f,
				.width = float(screenW),
				.height = float(screenH),
				.minDepth = 0,
				.maxDepth = 1,
			};
			const VkRect2D scissorRegion = {
				.offset = {0, 0},
				.extent = {screenW, screenH}
			};
			const VkPipelineViewportStateCreateInfo viewportInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.pViewports = &viewport,
				.scissorCount = 1,
				.pScissors = &scissorRegion,
			};

			const VkPipelineRasterizationStateCreateInfo rasterizationInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.depthClampEnable = VK_FALSE,
				.rasterizerDiscardEnable = VK_FALSE,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
				.depthBiasEnable = VK_FALSE,
				//.depthBiasConstantFactor = 0,
				//.depthBiasClamp = VK_FALSE,
				//.depthBiasSlopeFactor = 0,
				.lineWidth = 1.f,
			};

			const VkPipelineMultisampleStateCreateInfo multisampleInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
				.sampleShadingEnable = VK_FALSE,
			};

			const VkPipelineColorBlendAttachmentState blendAttachment = { .blendEnable = VK_FALSE, .colorWriteMask = VK_COLOR_COMPONENT_RGBA_BITS };
			const VkPipelineColorBlendStateCreateInfo blendInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &blendAttachment
			};


			{
				const VkDescriptorSetLayoutBinding binding = {
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1, // world be more than 1 for arrays of buffers
					.stageFlags = VK_SHADER_STAGE_ALL
				};
				const VkDescriptorSetLayoutCreateInfo info = {
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.bindingCount = 1,
					.pBindings = &binding,
				};
				vkRes = vkCreateDescriptorSetLayout(vk.device, &info, nullptr, &vk.descriptorSetLayout);
				assert(vkRes == VK_SUCCESS);
			}

			VkPipelineLayoutCreateInfo layoutInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &vk.descriptorSetLayout,
				.pushConstantRangeCount = 0,
				.pPushConstantRanges = nullptr
			};
			vkRes = vkCreatePipelineLayout(vk.device, &layoutInfo, nullptr, &vk.pipelineLayout);
			assert(vkRes == VK_SUCCESS);

			const VkDynamicState dynamicStates[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = std::size(dynamicStates),
				.pDynamicStates = dynamicStates
			};

			VkGraphicsPipelineCreateInfo info{ .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = std::size(shaderStages),
				.pStages = shaderStages,
				.pVertexInputState = &vertexInputInfo,
				.pInputAssemblyState = &inputAssemblyInfo,
				.pTessellationState = nullptr,
				.pViewportState = &viewportInfo,
				.pRasterizationState = &rasterizationInfo,
				.pMultisampleState = &multisampleInfo,
				.pDepthStencilState = nullptr,
				.pColorBlendState = &blendInfo,
				.pDynamicState = &dynamicState,
				.layout = vk.pipelineLayout,
				.renderPass = vk.renderPass, // this is a reference renderPass, but this pipeline can be used with compatible renderPasses
				.subpass = 0,
				.basePipelineHandle = VK_NULL_HANDLE,
				.basePipelineIndex = 0,
			};
			vkRes = vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &info, nullptr, &vk.pipeline);
			assert(vkRes == VK_SUCCESS);
		}
	}

	createSwapchainAndFramebuffers(screenW, screenH);

	{ // -- create vertex buffer
		const VkBufferCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.flags = 0,
			.size = sizeof(Vert) * 3,
			.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		vkRes = vkCreateBuffer(vk.device, &info, nullptr, &vk.vertexBuffer);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- alloc memory for the vertex buffer
		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(vk.device, vk.vertexBuffer, &memReqs);
		u32 memoryTypeToAllocateFrom = -1;
		for (u32 memTypeInd = 0; memTypeInd < vk.deviceMemProps.memoryTypeCount; memTypeInd++) {
			auto& memType = vk.deviceMemProps.memoryTypes[memTypeInd];
			if ( (memReqs.memoryTypeBits & (1 << memTypeInd) ) &&
				 (memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			{
				memoryTypeToAllocateFrom = memTypeInd;
				break;
			}
		}

		const auto nonCoherentAtomSize = vk.physicalDeviceProps.limits.nonCoherentAtomSize; // is the size and alignment in bytes that bounds concurrent access to host-mapped device memory.
		VkMemoryAllocateInfo allocInfo = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memReqs.size,
			.memoryTypeIndex = memoryTypeToAllocateFrom
		};
		vkRes = vkAllocateMemory(vk.device, &allocInfo, nullptr, &vk.vertexBufferMemory);
		assert(vkRes == VK_SUCCESS);

		// -- bind the memory to the buffer
		vkRes = vkBindBufferMemory(vk.device, vk.vertexBuffer, vk.vertexBufferMemory, 0);
		assert(vkRes == VK_SUCCESS);

		// -- upload the data to the vertex buffer
		void* data;
		vkRes = vkMapMemory(vk.device, vk.vertexBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
		assert(vkRes == VK_SUCCESS);
		memcpy(data, triangle, sizeof(triangle));

		const VkMappedMemoryRange range{
			.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
			.memory = vk.vertexBufferMemory,
			.offset = 0,
			.size = VK_WHOLE_SIZE
			// https://stackoverflow.com/questions/69181252/the-proper-way-to-invalidate-and-flush-vulkan-memory
		};
		vkRes = vkFlushMappedMemoryRanges(vk.device, 1, &range); // tell vulkan which parts of the memory we have modified
		assert(vkRes == VK_SUCCESS);

		vkUnmapMemory(vk.device, vk.vertexBufferMemory); // when we don't need the mapped pointer anymore we call unmap.
			//However, we could just keep the pointer around, and there shoudn't be any performance penalty

	}

	{ // -- create descriptor pool
		const VkDescriptorPoolSize sizes[] = { {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2,
		} };
		const VkDescriptorPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 2,
			.poolSizeCount = sizeof(sizes),
			.pPoolSizes = sizes
		};
		vkRes = vkCreateDescriptorPool(vk.device, &info, nullptr, &vk.descriptorPool);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- uniform buffers
		const VkBufferCreateInfo bufferInfo = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(Ubo),
			.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		for(int i = 0; i < 2; i++)
			vkCreateBuffer(vk.device, &bufferInfo, nullptr, vk.ubos + i);

		VkMemoryRequirements memReqs;
		vkGetBufferMemoryRequirements(vk.device, vk.ubos[0], &memReqs);
		const size_t memPerBuffer = (memReqs.size + memReqs.alignment - 1) / memReqs.alignment * memReqs.alignment;

		u32 memoryTypeToAllocateFrom = -1;
		for (u32 memTypeInd = 0; memTypeInd < vk.deviceMemProps.memoryTypeCount; memTypeInd++) {
			auto& memType = vk.deviceMemProps.memoryTypes[memTypeInd];
			if ((memReqs.memoryTypeBits & (1 << memTypeInd)) &&
				(memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
			{
				memoryTypeToAllocateFrom = memTypeInd;
				break;
			}
		}
		
		const VkMemoryAllocateInfo allocInfo = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memPerBuffer * std::size(vk.ubos),
			.memoryTypeIndex = memoryTypeToAllocateFrom,
		};
		vkRes = vkAllocateMemory(vk.device, &allocInfo, nullptr, &vk.ubosMemory);
		assert(vkRes == VK_SUCCESS);

		for (size_t i = 0; i < 2; i++) {
			vkRes = vkBindBufferMemory(vk.device, vk.ubos[i], vk.ubosMemory, i* memPerBuffer);
			assert(vkRes == VK_SUCCESS);
		}

		void* data;
		vkRes = vkMapMemory(vk.device, vk.ubosMemory, 0, VK_WHOLE_SIZE, 0, &data);
		assert(vkRes == VK_SUCCESS);
		for (size_t i = 0; i < 2; i++) {
			Ubo* dataB = (Ubo*)((char*)data + i * memPerBuffer);
			dataB->color = { 0, 0, 1 };
		}
		const VkMappedMemoryRange range{
			.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
			.memory = vk.vertexBufferMemory,
			.offset = 0,
			.size = VK_WHOLE_SIZE
			// https://stackoverflow.com/questions/69181252/the-proper-way-to-invalidate-and-flush-vulkan-memory
		};
		vkRes = vkFlushMappedMemoryRanges(vk.device, 1, &range); // tell vulkan which parts of the memory we have modified
		assert(vkRes == VK_SUCCESS);

		vkUnmapMemory(vk.device, vk.vertexBufferMemory); // when we don't need the mapped pointer anymore we call unmap.
			// However, we could just keep the pointer around, and there shoudn't be any performance penalty
	}

	{ // allocate descriptor sets
		const VkDescriptorSetLayout layouts[] = {vk.descriptorSetLayout, vk.descriptorSetLayout};
		VkDescriptorSetAllocateInfo info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = vk.descriptorPool,
			.descriptorSetCount = 2,
			.pSetLayouts = layouts,
		};
		vkRes = vkAllocateDescriptorSets(vk.device, &info, vk.descriptorSets);
		assert(vkRes == VK_SUCCESS);
	}

	// fill descriptor sets
	for (int i = 0; i < 2; i++) {
		VkDescriptorBufferInfo bufferInfo = {
			.buffer = vk.ubos[i],
			.offset = 0,
			.range = VK_WHOLE_SIZE
		};

		VkWriteDescriptorSet writeInfo = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk.descriptorSets[i],
			.dstBinding = 0,
			.dstArrayElement = 0, // dstArrayElement and descriptorCount indicate the subrange of the array
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &bufferInfo,
		};
		vkUpdateDescriptorSets(vk.device, 1, &writeInfo, 0, nullptr);
	}

	{ // -- create command pools
		const VkCommandPoolCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			//.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = vk.graphicsQueueFamily,
		};
		vkRes = vkCreateCommandPool(vk.device, &info, nullptr, &vk.graphicsCmdPool);
		assert(vkRes == VK_SUCCESS);
	}

	// -- create cmd buffers
	allocateCmdBuffers();

	// -- record cmd buffers
	recordCmdBuffers(screenW, screenH);

	// -- create semaphores
	createSemaphores();

	{ // -- create fences
		const VkFenceCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};
		for (int i = 0; i < 2; i++) {
			vkRes = vkCreateFence(vk.device, &info, nullptr, &vk.fence_queueWorkFinished[i]);
			assert(vkRes == VK_SUCCESS);
		}
	}

	// --- main loop ---
	u32 frameInd = 0;
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height)
		{
			//printf("R\n");
			vkDeviceWaitIdle(vk.device);

			createSwapchainAndFramebuffers(width, height);
			createSemaphores();

			vkResetCommandPool(vk.device, vk.graphicsCmdPool, 0);
			recordCmdBuffers(width, height);

			//frameInd = 0;
		});

		u32 swapchainImgInd;
		vkRes = vkAcquireNextImageKHR(vk.device, vk.swapchain,
			u64(-1), // timeout
			vk.semaphore_swapchainImgAvailable[frameInd], // semaphore to signal
			VK_NULL_HANDLE, // fence to signal
			&swapchainImgInd);
		assert(vkRes == VK_SUCCESS);

		vkRes = vkWaitForFences(vk.device, 1, &vk.fence_queueWorkFinished[frameInd], VK_TRUE, u64(-1));
		assert(vkRes == VK_SUCCESS);
		vkResetFences(vk.device, 1, &vk.fence_queueWorkFinished[frameInd]);
		assert(vkRes == VK_SUCCESS);

		const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		const VkSubmitInfo submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &vk.semaphore_swapchainImgAvailable[frameInd],
			.pWaitDstStageMask = &waitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &vk.cmdBuffers[swapchainImgInd],
				// cmd buffers are recorded with a specific framebuffer(which points to a specific img view of the swapchain).
				// That's why we use swapchaimImgInd here instad of frameInd
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &vk.semaphore_drawFinished[frameInd]
		};
		vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, vk.fence_queueWorkFinished[frameInd]);

		const VkPresentInfoKHR presentInfo = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &vk.semaphore_drawFinished[frameInd],
			.swapchainCount = 1,
			.pSwapchains = &vk.swapchain,
			.pImageIndices = &swapchainImgInd,
			.pResults = &vkRes
		};
		vkQueuePresentKHR(vk.graphicsQueue, &presentInfo);
		assert(vkRes == VK_SUCCESS);

		frameInd = (frameInd + 1) % 2;
	}
}