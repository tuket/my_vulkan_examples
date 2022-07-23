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

struct Vk {
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	u32 graphicsQueueFamily;
	VkQueue graphicsQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	VkImageView swapchainImageViews[2];
	VkFramebuffer framebuffers[2];
	VkRenderPass renderPass;
	VkPipeline pipeline;
	VkCommandPool graphicsCmdPool;
	VkCommandBuffer graphicsCmdBuffers[2];
	VkSemaphore semaphore_swapchainImgAvailable[2];
	VkSemaphore semaphore_drawFinished[2];
	VkFence fence_queueWorkFinished[2];
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

static char* loadTextFile(const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
		return nullptr;
	fseek(file, 0, SEEK_END);
	const long len = ftell(file);
	fseek(file, 0, SEEK_SET);
	char* s = new char[len + 1];
	fread(s, 1, len, file);
	s[len] = '\0';
	fclose(file);
	return s;
}

const auto VK_COLOR_COMPONENT_RGBA_BITS = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

/*VkShaderModule loadGlslShaderModule(const char* fileName)
{
	const auto src = loadTextFileL(fileName);
	if (!src.str) {
		printf("Error loading shader: %s\n", fileName);
		exit(-1);
	}

	assert(shaderCompiler);
	shaderc_compile_options_t options = nullptr;
	const auto res = shaderc_compile_into_spv(shaderCompiler, src.str, src.len, shaderc_shader_kind::shaderc_vertex_shader, fileName, "main", options);
	const size_t numErrors = shaderc_result_get_num_errors(res);
	if (numErrors) {
		printf("Error compiling '%s':\n", fileName);
		printf("%.*s\n%s\n", src.len, src.str, shaderc_result_get_error_message(res));
		exit(-1);
	}

	const size_t spvSize = shaderc_result_get_length(res);
	const auto spv = (const u32*)shaderc_result_get_bytes(res);

	VkShaderModule module;
	const VkShaderModuleCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = spvSize,
		.pCode = spv
	};
	vkCreateShaderModule(vk.device, &info, nullptr, &module);

	shaderc_result_release(res);
	delete[] src.str;

	return module;
}*/

VkShaderModule loadSpirvShaderModule(const char* fileName)
{
	Buffer fileBuffer = loadBinaryFile(fileName);
	if (!fileBuffer.data) {
		printf("Error loading shader: %s\n", fileName);
		exit(-1);
	}

	const u32* spv = (u32*)fileBuffer.data;
	const VkShaderModuleCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = fileBuffer.len,
		.pCode = spv
	};
	VkShaderModule shaderModule;
	vkCreateShaderModule(vk.device, &info, nullptr, &shaderModule);

	delete[] fileBuffer.data;
	return shaderModule;
}

static VkPipelineShaderStageCreateInfo makeStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule module)
{
	VkPipelineShaderStageCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	info.stage = stage;
	info.module = module;
	info.pName = "main";
	info.pSpecializationInfo = nullptr; // allows to specify values for shader constants
	return info;
}

void example_0()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // explicitly tell GLFW not to create an OpenGL when creating the window
	window = glfwCreateWindow(800, 600, "vulkan example", nullptr, nullptr);

	VkResult vkRes;
	{ // -- create vulkan instance
		u32 numRequiredExtensions;
		const char** const requiredExtensions = glfwGetRequiredInstanceExtensions(&numRequiredExtensions);

		const VkApplicationInfo appInfo{ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = "example",
			.apiVersion = VK_API_VERSION_1_0
		};
		const VkInstanceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = 0,
			.ppEnabledLayerNames = nullptr,
			.enabledExtensionCount = numRequiredExtensions,
			.ppEnabledExtensionNames = requiredExtensions,
		};
		vkRes = vkCreateInstance(&info, nullptr, &vk.instance);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create device
		constexpr u32 MAX_PHYSICAL_DEVICES = 8;
		VkPhysicalDevice physicalDevices[MAX_PHYSICAL_DEVICES];
		u32 numPhysicalDevices = MAX_PHYSICAL_DEVICES;
		vkRes = vkEnumeratePhysicalDevices(vk.instance, &numPhysicalDevices, physicalDevices);
		assert(vkRes == VK_SUCCESS && numPhysicalDevices > 0);

		VkPhysicalDeviceProperties props[MAX_PHYSICAL_DEVICES];
		VkPhysicalDeviceMemoryProperties memProps[MAX_PHYSICAL_DEVICES];
		for (u32 i = 0; i < numPhysicalDevices; i++) {
			vkGetPhysicalDeviceProperties(physicalDevices[i], &props[i]);
			vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProps[i]);
		}

		// choose best phyiscal device
		auto comparePhysicalDevices = [&](u32 a, u32 b) -> i64
		{
			auto deviceTypeScore = [](VkPhysicalDeviceType t) -> u32 {
				switch (t) {
				case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
				case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
				case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
				default: return 0;
				}
			};
			const u32 deviceTypeScoreA = deviceTypeScore(props[a].deviceType);
			const u32 deviceTypeScoreB = deviceTypeScore(props[b].deviceType);
			if (deviceTypeScoreA != deviceTypeScoreB)
				return deviceTypeScoreA - deviceTypeScoreB;

			auto calcMem = [](const VkPhysicalDeviceMemoryProperties& a) -> u64
			{
				u64 mem = 0;
				for (u32 i = 0; i < a.memoryHeapCount; i++) {
					if (a.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
						mem = glm::max(mem, a.memoryHeaps[i].size);
				}
				return mem;
			};
			const u64 memA = calcMem(memProps[a]);
			const u64 memB = calcMem(memProps[b]);
			return i64(memA) - i64(memB);
		};

		u32 bestPhysicalDevice = 0;
		for (u32 i = 1; i < numPhysicalDevices; i++) {
			if (comparePhysicalDevices(i, bestPhysicalDevice) > 0)
				bestPhysicalDevice = i;
		}
		vk.physicalDevice = physicalDevices[bestPhysicalDevice];

		vkRes = glfwCreateWindowSurface(vk.instance, window, nullptr, &vk.surface);
		assert(vkRes == VK_SUCCESS);

		constexpr u32 MAX_QUEUE_FAMILIES = 8;
		u32 numQueueFamilies = MAX_QUEUE_FAMILIES;
		VkQueueFamilyProperties queueFamilyProps[MAX_QUEUE_FAMILIES];
		vkGetPhysicalDeviceQueueFamilyProperties(vk.physicalDevice, &numQueueFamilies, queueFamilyProps);

		vk.graphicsQueueFamily = numQueueFamilies;
		for (u32 i = 0; i < numQueueFamilies; i++) {
			const bool supportsGraphics = queueFamilyProps[0].queueFlags & VK_QUEUE_GRAPHICS_BIT;
			VkBool32 supportsPresentation;
			// in real life HW there is no driver that provides graphics and presentation as separate families: https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
			vkGetPhysicalDeviceSurfaceSupportKHR(vk.physicalDevice, i, vk.surface, &supportsPresentation);
			if (supportsGraphics && supportsPresentation) {
				vk.graphicsQueueFamily = i;
				break;
			}
		}
		assert(vk.graphicsQueueFamily != numQueueFamilies);

		const float queuePriorities[] = { 1.f };
		VkDeviceQueueCreateInfo queueCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = vk.graphicsQueueFamily,
			.queueCount = 1,
			.pQueuePriorities = queuePriorities,
		};

		const char* extensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		const VkDeviceCreateInfo deviceInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queueCreateInfo,
			.enabledExtensionCount = std::size(extensionNames),
			.ppEnabledExtensionNames = extensionNames
			// enabledLayerCount, ppEnabledLayerNames, pEnabledFeatures
		};
		vkRes = vkCreateDevice(vk.physicalDevice, &deviceInfo, nullptr, &vk.device);
		assert(vkRes == VK_SUCCESS);

		vkGetDeviceQueue(vk.device, vk.graphicsQueueFamily, 0, &vk.graphicsQueue);
		// queues
		// https://community.khronos.org/t/guidelines-for-selecting-queues-and-families/7222
		// https://www.reddit.com/r/vulkan/comments/aara8f/best_way_for_selecting_queuefamilies/
		// https://stackoverflow.com/questions/37575012/should-i-try-to-use-as-many-queues-as-possible
	}

	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &surfaceCapabilities);
	printf("min images: %d\nmax images: %d\n", surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount);

	const u32 screenW = surfaceCapabilities.currentExtent.width;
	const u32 screenH = surfaceCapabilities.currentExtent.height;
	{ // -- create swapchain
		VkSwapchainCreateInfoKHR info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = vk.surface,
			.minImageCount = 2,
			.imageFormat = VK_FORMAT_B8G8R8A8_SRGB,
			.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, // https://stackoverflow.com/questions/66401081/vulkan-swapchain-format-unorm-vs-srgb
			.imageExtent = {screenW, screenH},
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, // only once queue will reference an image at the same time. We can still reference the image from different queues, but not at the same time. We must use a memory barrier for this!
			.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_MAILBOX_KHR, // with 2 images, MAILBOX and FIFO are equivalent
			.clipped = VK_TRUE, // this allows to discard the rendering of hidden pixel regions. E.g a window is partially covered by another
			//.oldSwapchain =,
		};
		vkRes = vkCreateSwapchainKHR(vk.device, &info, nullptr, &vk.swapchain);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create image views of the swapchain
		VkImage images[2];
		u32 numImages = std::size(images);
		vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &numImages, nullptr);
		assert(numImages == 2);
		vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &numImages, images);

		for (u32 i = 0; i < 2; i++) {
			const VkImageViewCreateInfo info{ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = images[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_SRGB,
				//info.components = VK_COMPONENT_SWIZZLE_IDENTITY,
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT ,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				}
			};
			vkCreateImageView(vk.device, &info, nullptr, vk.swapchainImageViews + i);
		}
	}

	{ // -- create render pass
		const VkAttachmentDescription attachment = {
			.flags = 0,
			.format = VK_FORMAT_B8G8R8A8_SRGB,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, // clear at the beginning of the first subpass where it's used
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE, // keep the stored value, there might be depth attachments that don't need persistance
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, // we don't care to preserve contents
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		};
		VkAttachmentReference attachmentRef = {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
		const VkSubpassDescription subpass = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &attachmentRef,
		};
		VkRenderPassCreateInfo info = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
		info.attachmentCount = 1;
		info.pAttachments = &attachment;
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 0;
		info.pDependencies = nullptr;
		vkRes = vkCreateRenderPass(vk.device, &info, nullptr, &vk.renderPass);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create graphics pipeline
		VkShaderModule vertShad = loadSpirvShaderModule(SHADERS_FOLDER"simple_vert.spirv");
		VkShaderModule fragShad = loadSpirvShaderModule(SHADERS_FOLDER"simple_frag.spirv");

		const VkPipelineShaderStageCreateInfo stagesInfos[] = {
			makeStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShad),
			makeStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShad)
		};

		VkPipelineVertexInputStateCreateInfo vertInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
		vertInputInfo.vertexBindingDescriptionCount = 0;
		vertInputInfo.pVertexBindingDescriptions = nullptr;
		vertInputInfo.vertexAttributeDescriptionCount = 0;
		vertInputInfo.pVertexAttributeDescriptions = nullptr;

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
		inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

		const VkViewport viewport = {
			.x = 0, .y = 0,
			.width = float(screenW), .height = float(screenH),
			.minDepth = 0, .maxDepth = 1
		};
		const VkRect2D scissor = { {0, 0}, surfaceCapabilities.currentExtent };
		VkPipelineViewportStateCreateInfo viewportInfo = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterInfo = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
		rasterInfo.depthClampEnable = VK_FALSE; // according to vulkan-tutorial this can be useful when rendering shadowmaps but I haven't found any reference
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1;

		VkPipelineMultisampleStateCreateInfo multisampleInfo = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
		multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleInfo.sampleShadingEnable = VK_FALSE;

		//VkPipelineDepthStencilStateCreateInfo depthStencilInfo = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };

		VkPipelineColorBlendAttachmentState blendAttchment = {};
		blendAttchment.blendEnable = VK_FALSE;
		blendAttchment.colorWriteMask = VK_COLOR_COMPONENT_RGBA_BITS;
		VkPipelineColorBlendStateCreateInfo blendInfo = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = &blendAttchment;

		VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
		layoutInfo.setLayoutCount = 0;
		layoutInfo.pSetLayouts = nullptr;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;
		VkPipelineLayout layout;
		vkRes = vkCreatePipelineLayout(vk.device, &layoutInfo, nullptr, &layout);
		assert(vkRes == VK_SUCCESS);

		VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
		pipelineInfo.stageCount = std::size(stagesInfos);
		pipelineInfo.pStages = stagesInfos;
		pipelineInfo.pVertexInputState = &vertInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportInfo;
		pipelineInfo.pRasterizationState = &rasterInfo;
		pipelineInfo.pMultisampleState = &multisampleInfo;
		//pipelineInfo.pDepthStencilState = &depthStencilInfo;
		pipelineInfo.pColorBlendState = &blendInfo;
		//pipelineInfo.pDynamicState = ; // TODO: dynamic state for viewport and scissor at least
		pipelineInfo.layout = layout;
		pipelineInfo.renderPass = vk.renderPass;
		pipelineInfo.subpass = 0;
		vkRes = vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk.pipeline);
		assert(vkRes == VK_SUCCESS);
	}

	// -- create framebuffers
	for (int i = 0; i < 2; i++)
	{
		VkFramebufferCreateInfo info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
		info.renderPass = vk.renderPass; // the FB will be compatible with renderPasses similar to this: https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#renderpass-compatibility
		info.attachmentCount = 1;
		info.pAttachments = &vk.swapchainImageViews[i];
		info.width = screenW;
		info.height = screenH;
		info.layers = 1;
		vkCreateFramebuffer(vk.device, &info, nullptr, &vk.framebuffers[i]);
	}

	{ // -- create cmd pools
		VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		info.flags = 0;
		info.queueFamilyIndex = vk.graphicsQueueFamily;
		vkRes = vkCreateCommandPool(vk.device, &info, nullptr, &vk.graphicsCmdPool);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create cmd buffers
		VkCommandBufferAllocateInfo info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		info.commandPool = vk.graphicsCmdPool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 2;
		vkAllocateCommandBuffers(vk.device, &info, vk.graphicsCmdBuffers);
	}

	for (int i = 0; i < 2; i++) { // -- record cmd buffers
		const VkCommandBufferBeginInfo info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		vkRes = vkBeginCommandBuffer(vk.graphicsCmdBuffers[i], &info);
		assert(vkRes == VK_SUCCESS);

		const VkClearValue CLEAR_VALUE = { .color = {.float32 = {0.2f, 0.2f, 0.2f, 0.f}} };
		const VkRenderPassBeginInfo rpInfo = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = vk.renderPass,
			.framebuffer = vk.framebuffers[i],
			.renderArea = { {0, 0}, {screenW, screenH} },
			.clearValueCount = 1,
			.pClearValues = &CLEAR_VALUE,
		};
		vkCmdBeginRenderPass(vk.graphicsCmdBuffers[i], &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(vk.graphicsCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipeline);

		// draw the triangle!
		vkCmdDraw(vk.graphicsCmdBuffers[i], 3, 1, 0, 0);

		vkCmdEndRenderPass(vk.graphicsCmdBuffers[i]);

		vkRes = vkEndCommandBuffer(vk.graphicsCmdBuffers[i]);
		assert(vkRes == VK_SUCCESS);
	}

	{ // -- create semaphores
		const VkSemaphoreCreateInfo info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		for (int i = 0; i < 2; i++)
			vkCreateSemaphore(vk.device, &info, nullptr, &vk.semaphore_swapchainImgAvailable[i]);
		for (int i = 0; i < 2; i++)
			vkCreateSemaphore(vk.device, &info, nullptr, &vk.semaphore_drawFinished[i]);
	}

	{ // -- create fences
		const VkFenceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};
		for (int i = 0; i < 2; i++)
			vkCreateFence(vk.device, &info, nullptr, &vk.fence_queueWorkFinished[i]);
	}

	// --- loop ---
	u32 frameInd = 0;
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		const u64 SECONDS_TO_NANOSECODS = 1'000'000'000;
		u32 imgInd = 0;
		vkAcquireNextImageKHR(vk.device, vk.swapchain,
			5 * SECONDS_TO_NANOSECODS, // timeout
			vk.semaphore_swapchainImgAvailable[frameInd], // semaphore to signal
			VK_NULL_HANDLE, // fence to signal
			&imgInd);

		vkWaitForFences(vk.device, 1, &vk.fence_queueWorkFinished[frameInd], VK_TRUE, -1);
		vkResetFences(vk.device, 1, &vk.fence_queueWorkFinished[frameInd]);

		const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		const VkSubmitInfo submitInfo = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &vk.semaphore_swapchainImgAvailable[frameInd],
			.pWaitDstStageMask = &waitStage,
			.commandBufferCount = 1,
			.pCommandBuffers = &vk.graphicsCmdBuffers[frameInd],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &vk.semaphore_drawFinished[frameInd]
		};
		vkRes = vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, vk.fence_queueWorkFinished[frameInd]);

		const VkPresentInfoKHR presentInfo{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &vk.semaphore_drawFinished[frameInd],
			.swapchainCount = 1,
			.pSwapchains = &vk.swapchain,
			.pImageIndices = &imgInd,
			.pResults = &vkRes
		};
		vkQueuePresentKHR(vk.graphicsQueue, &presentInfo);
		assert(vkRes == VK_SUCCESS);

		frameInd = (frameInd + 1) % 2;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}