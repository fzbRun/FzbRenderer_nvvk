#include <nvapp/application.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/context.hpp>
#include "common/Application/Application.h"
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>


int main(int argc, char** argv) {
    nvutils::ParameterParser cli(nvutils::getExecutablePath().stem().string());   //将可执行文件名作为参数传入
    //nvutils::ParameterRegistry reg;
    //reg.add({ "headless", "Run in headless mode" }, &appInfo.headless, true);   //headless表示是否不要窗口
    //cli.add(reg);
    cli.parse(argc, argv);

    nvapp::ApplicationCreateInfo appInfo{};
    nvvk::Context vkContext;
    auto fzbRenderer_nvvk = std::make_shared<FzbRenderer::Application>(appInfo, vkContext);

    appInfo.vSync = false;

    nvapp::Application application;
    application.init(appInfo);

    auto elemCamera = std::make_shared<nvapp::ElementCamera>();
    auto windowTitle = std::make_shared<nvapp::ElementDefaultWindowTitle>();
    auto windowMenu = std::make_shared<nvapp::ElementDefaultMenu>();
    auto camManip = fzbRenderer_nvvk->getCameraManipulator();
    elemCamera->setCameraManipulator(camManip);

    application.addElement(windowMenu);
    application.addElement(windowTitle);
    application.addElement(elemCamera);
    application.addElement(fzbRenderer_nvvk);

    application.run();
    application.deinit();
    vkContext.deinit();

    return 0;
}
