#pragma once

#include "donut/engine/Scene.h"

struct GeometryData;
struct InstanceData;

namespace donut::engine
{
    class OptiXScene : public Scene
    {
    public:
        OptiXScene(
            nvrhi::IDevice* device,
            ShaderFactory& shaderFactory,
            const std::shared_ptr<vfs::IFileSystem>& fs,
            const std::shared_ptr<TextureCache>& textureCache,
            const std::shared_ptr<DescriptorTableManager>& descriptorTable,
            const std::shared_ptr<SceneTypeFactory>& sceneTypeFactory)
            : Scene(device, shaderFactory, fs, textureCache, descriptorTable, sceneTypeFactory)
        {
        }

    protected:
        template<typename T>
        static void createDescriptor(
            const DescriptorHandle& descriptor,
            T ptr,
            std::vector<T>& Space);

        void createTextureDescriptor(
            std::shared_ptr<LoadedTexture> texture);

        void CreateMeshBuffers(nvrhi::ICommandList* commandList) override;
        nvrhi::BufferHandle CreateMaterialBuffer() override;
        nvrhi::BufferHandle CreateGeometryBuffer() override;
        nvrhi::BufferHandle CreateInstanceBuffer() override;
        nvrhi::BufferHandle CreateMaterialConstantBuffer(const std::string& debugName) override;
        void UpdateMaterial(const std::shared_ptr<Material>& material) override;

        nvrhi::OptiXTraversableHandle handle;

        std::vector<CUdeviceptr> BindlessBuffers;
        std::vector<CUtexObject> BindlessTextures;

    public:
        CUdeviceptr* d_BindlessBuffers = 0;
        CUtexObject* d_BindlessTextures = 0;

        InstanceData* t_InstanceData;
        GeometryData* t_GeometryData;
        MaterialConstants* t_MaterialConstants;

        nvrhi::OptiXTraversableHandle buildOptixAccelStructs();

        [[nodiscard]] nvrhi::OptiXTraversableHandle GetTraversableHandle() const
        {
            return handle;
        }
    };

    template<typename T>
    void OptiXScene::createDescriptor(
        const DescriptorHandle& descriptor,
        T ptr,
        std::vector<T>& Space)
    {
        auto table_idx = descriptor.Get();
        if (Space.size() < table_idx + 1)
        {
            Space.resize(table_idx + 1, 0);
        }
        Space[table_idx] = ptr;
    }

} // namespace donut::engine
