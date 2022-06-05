#include <stdio.h>
#include <assert.h>
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;

#define SHADERS_FOLDER "shaders/"

struct Vk {
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapchain;
	VkImageView swapchainImageViews[2];
};
static Vk vk;
static GLFWwindow* window;
static auto shaderCompiler = shaderc_compiler_initialize();

struct StrL {
	u32 len;
	char* str;
};
static StrL loadTextFileL(const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
		return {};
	fseek(file, 0, SEEK_SET);
	const u32 len = ftell(file);
	char* s = new char[len];
	fread(s, 1, len, file);
	s[len] = '\0';
	fclose(file);
	return {len, s};
}

static char* loadTextFile(const char* fileName)
{
	FILE* file = fopen(fileName, "r");
	if (!file)
		return nullptr;
	fseek(file, 0, SEEK_SET);
	const long len = ftell(file);
	char* s = new char[len + 1];
	fread(s, 1, len, file);
	s[len] = '\0';
	fclose(file);
	return s;
}

VkShaderModule loadGlslShaderModule(const char* fileName)
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
}

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // explicitly tell GLFW not to create an OpenGL when creating the window
	window = glfwCreateWindow(800, 600, "vulkan example", nullptr, nullptr);

	VkResult vkRes;
	{ // create vulkan instance
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

	{ // create device
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

		u32 graphicsQueueFamily = numQueueFamilies;
		u32 presentationQueueFamily = numQueueFamilies;
		for (u32 i = 0; i < numQueueFamilies; i++) {
			const bool supportsGraphics = queueFamilyProps[0].queueFlags & VK_QUEUE_GRAPHICS_BIT;
			VkBool32 supportsPresentation;
			vkGetPhysicalDeviceSurfaceSupportKHR(vk.physicalDevice, i, vk.surface, &supportsPresentation);
			if (supportsGraphics && supportsPresentation) {
				graphicsQueueFamily = presentationQueueFamily = i;
			}
			else {
				if (graphicsQueueFamily == numQueueFamilies && supportsGraphics)
					graphicsQueueFamily = i;
				if (presentationQueueFamily == numQueueFamilies && supportsPresentation)
					presentationQueueFamily = i;
			}
		}
		assert(graphicsQueueFamily != numQueueFamilies);
		assert(presentationQueueFamily != numQueueFamilies);

		auto initQueueCreateInfo = [&](VkDeviceQueueCreateInfo& info, u32 familyInd)
		{
			const float queuePriorities[] = { 1.f };
			info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
			info.queueCount = 1;
			info.queueFamilyIndex = familyInd;
			info.pQueuePriorities = queuePriorities;
		};
		u32 numQueueCreateInfos = 1;
		VkDeviceQueueCreateInfo queueCreateInfos[2];
		initQueueCreateInfo(queueCreateInfos[0], graphicsQueueFamily);
		if (graphicsQueueFamily != presentationQueueFamily) {
			numQueueCreateInfos = 2;
			initQueueCreateInfo(queueCreateInfos[1], presentationQueueFamily);
		}

		const char* extensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		const VkDeviceCreateInfo deviceInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.queueCreateInfoCount = numQueueCreateInfos,
			.pQueueCreateInfos = queueCreateInfos,
			.enabledExtensionCount = std::size(extensionNames),
			.ppEnabledExtensionNames = extensionNames
			// enabledLayerCount, ppEnabledLayerNames, pEnabledFeatures
		};
		vkRes = vkCreateDevice(vk.physicalDevice, &deviceInfo, nullptr, &vk.device);
		assert(vkRes == VK_SUCCESS);

		// queues
		// https://community.khronos.org/t/guidelines-for-selecting-queues-and-families/7222
		// https://www.reddit.com/r/vulkan/comments/aara8f/best_way_for_selecting_queuefamilies/
		// https://stackoverflow.com/questions/37575012/should-i-try-to-use-as-many-queues-as-possible
		//vkCreateDevice();
	}

	{ // create swapchain
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physicalDevice, vk.surface, &surfaceCapabilities);
		printf("min images: %d\nmax images: %d\n", surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount);
		VkSwapchainCreateInfoKHR info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = vk.surface,
			.minImageCount = 2,
			.imageFormat = VK_FORMAT_R8G8B8A8_SRGB,
			.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, // https://stackoverflow.com/questions/66401081/vulkan-swapchain-format-unorm-vs-srgb
			.imageExtent = surfaceCapabilities.currentExtent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, // only once queue will reference an image at the same time. We can still reference the image for different queues, but not at the same time. We must use a memory barrier for this!
			.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_MAILBOX_KHR, // with 2 images, MAILBOX and FIFO are equivalent
			.clipped = VK_TRUE, // this allows to discard the rendering of hidden pixel regions. E.g a window is partially covered by another
			//.oldSwapchain =,
		};
		vkRes = vkCreateSwapchainKHR(vk.device, &info, nullptr, &vk.swapchain);
		assert(vkRes == VK_SUCCESS);
	}

	{ // create image views of the swapchain
		VkImage images[2];
		u32 numImages = std::size(images);
		vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &numImages, nullptr);
		assert(numImages == 2);
		vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &numImages, images);

		for (u32 i = 0; i < 2; i++) {
			const VkImageViewCreateInfo info{ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = images[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_R8G8B8A8_SRGB,
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

	{ // create graphics pipeline
		VkShaderModule vertShad = loadGlslShaderModule(SHADERS_FOLDER"vert.glsl");

	}
}