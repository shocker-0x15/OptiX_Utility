/*

JP: このサンプルはインタラクティブなアプリケーション上で
    動的にオブジェクトを追加・削除・移動する方法の一例を示します。
    APIの使い方として新しいものを示すサンプルではありません。

EN: This sample demonstrates an example of dynamically add, remove and move objects
    in an interactive application.
    This sample has nothing new for API usage.

*/

#include "scene_edit_shared.h"

#include "../common/gui_common.h"
#include "../common/imgui_file_dialog.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"
#include "../common/dds_loader.h"



constexpr cudau::BufferType g_bufferType = cudau::BufferType::Device;

using VertexBufferRef = std::shared_ptr<cudau::TypedBuffer<Shared::Vertex>>;

struct OptiXEnv;
struct GeometryInstance;
struct GeometryGroup;
struct Instance;
struct Group;
using GeometryInstanceRef = std::shared_ptr<GeometryInstance>;
using GeometryInstanceWRef = std::weak_ptr<GeometryInstance>;
using GeometryGroupRef = std::shared_ptr<GeometryGroup>;
using GeometryGroupWRef = std::weak_ptr<GeometryGroup>;
using InstanceRef = std::shared_ptr<Instance>;
using InstanceWRef = std::weak_ptr<Instance>;
using GroupRef = std::shared_ptr<Group>;
using GroupWRef = std::weak_ptr<Group>;

struct GeometryInstance {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::GeometryInstance optixGeomInst;
    VertexBufferRef vertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> triangleBuffer;

    GeometryInstance() {}
    static void finalize(GeometryInstance* p);
};

struct GeometryInstanceFileGroup {
    uint32_t serialID;
    std::string name;
    std::vector<GeometryInstanceRef> geomInsts;
};

struct GeometryInstancePreTransform {
    float3 scale;
    float rollPitchYaw[3];
    float3 position;
};

struct GeometryGroup {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::GeometryAccelerationStructure optixGAS;
    std::vector<GeometryInstanceRef> geomInsts;
    std::vector<GeometryInstancePreTransform> preTransforms;
    cudau::TypedBuffer<std::array<float, 12>> preTransformBuffer;
    std::set<InstanceWRef, std::owner_less<InstanceWRef>> parentInsts;
    cudau::Buffer optixGasMem;

    GeometryGroup() {}
    static void finalize(GeometryGroup* p);

    void propagateMarkDirty() const;
};

struct Instance {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::Instance optixInst;
    GeometryGroupRef geomGroup;
    GroupRef group;
    std::set<GroupWRef, std::owner_less<GroupWRef>> parentGroups;
    float3 scale;
    float rollPitchYaw[3];
    float3 position;

    static void finalize(Instance* p);

    void propagateMarkDirty() const;
};

struct Group {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::InstanceAccelerationStructure optixIAS;
    std::vector<InstanceRef> insts;
    std::set<InstanceWRef, std::owner_less<InstanceWRef>> parentInsts;
    cudau::Buffer optixIasMem;
    cudau::TypedBuffer<OptixInstance> optixInstanceBuffer;

    static void finalize(Group* p);

    void propagateMarkDirty() const;
};

struct OptiXEnv {
    CUcontext cuContext;
    optixu::Context context;
    optixu::Material material;
    optixu::Scene scene;

    uint32_t geomInstSerialID;
    uint32_t geomInstFileGroupSerialID;
    uint32_t gasSerialID;
    uint32_t instSerialID;
    uint32_t iasSerialID;
    std::map<uint32_t, GeometryInstanceFileGroup> geomInstFileGroups;
    std::map<uint32_t, GeometryGroupRef> geomGroups;
    std::map<uint32_t, InstanceRef> insts;
    std::map<uint32_t, GroupRef> groups;

    cudau::Buffer asScratchBuffer;

    cudau::Buffer hitGroupSBT[2]; // double buffering
};

void GeometryInstance::finalize(GeometryInstance* p) {
    p->optixGeomInst.destroy();
    p->triangleBuffer.finalize();
    delete p;
}
void GeometryGroup::finalize(GeometryGroup* p) {
    p->optixGasMem.finalize();
    p->preTransformBuffer.finalize();
    p->optixGAS.destroy();
    delete p;
}
void GeometryGroup::propagateMarkDirty() const {
    for (const auto &parentWRef : parentInsts) {
        InstanceRef parent = parentWRef.lock();
        parent->propagateMarkDirty();
    }
}
void Instance::finalize(Instance* p) {
    p->optixInst.destroy();
    delete p;
}
void Instance::propagateMarkDirty() const {
    for (const auto &parentWRef : parentGroups) {
        GroupRef parent = parentWRef.lock();
        parent->propagateMarkDirty();
    }
}
void Group::finalize(Group* p) {
    p->optixInstanceBuffer.finalize();
    p->optixIasMem.finalize();
    p->optixIAS.destroy();
    delete p;
}
void Group::propagateMarkDirty() const {
    optixIAS.markDirty();
    for (const auto &parentWRef : parentInsts) {
        InstanceRef parent = parentWRef.lock();
        parent->propagateMarkDirty();
    }
}



void loadFile(const std::filesystem::path &filepath, CUstream stream, OptiXEnv* optixEnv) {
    Assimp::Importer importer;
    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80.0f);
    const aiScene* scene = importer.ReadFile(
        filepath.string(),
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_PreTransformVertices);
    if (!scene) {
        hpprintf("Failed to load %s.\n", filepath.string().c_str());
        return;
    }

    std::string basename = filepath.stem().string();

    GeometryInstanceFileGroup fileGroup;
    fileGroup.name = filepath.filename().string();

    for (int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx) {
        const aiMesh* mesh = scene->mMeshes[meshIdx];

        std::vector<Shared::Vertex> vertices(mesh->mNumVertices);
        for (int vIdx = 0; vIdx < mesh->mNumVertices; ++vIdx) {
            Shared::Vertex vtx;
            vtx.position = *reinterpret_cast<float3*>(&mesh->mVertices[vIdx]);
            vtx.normal = *reinterpret_cast<float3*>(&mesh->mNormals[vIdx]);
            if (mesh->mTextureCoords[0])
                vtx.texCoord = *reinterpret_cast<float2*>(&mesh->mTextureCoords[0][vIdx]);
            else
                vtx.texCoord = float2(0.0f, 0.0f);
            vertices[vIdx] = vtx;
        }

        std::vector<Shared::Triangle> triangles(mesh->mNumFaces);
        for (int fIdx = 0; fIdx < mesh->mNumFaces; ++fIdx) {
            const aiFace &face = mesh->mFaces[fIdx];

            Shared::Triangle tri;
            tri.index0 = face.mIndices[0];
            tri.index1 = face.mIndices[1];
            tri.index2 = face.mIndices[2];

            triangles[fIdx] = tri;
        }

        VertexBufferRef vertexBuffer = make_shared_with_deleter<cudau::TypedBuffer<Shared::Vertex>>(
            [](cudau::TypedBuffer<Shared::Vertex>* p) {
                p->finalize();
                delete p;
            });
        vertexBuffer->initialize(optixEnv->cuContext, g_bufferType, vertices, stream);

        char name[256];
        sprintf_s(name, "%s-%d", basename.c_str(), meshIdx);
        GeometryInstanceRef geomInst = make_shared_with_deleter<GeometryInstance>(GeometryInstance::finalize);
        geomInst->optixEnv = optixEnv;
        geomInst->serialID = optixEnv->geomInstSerialID++;
        geomInst->name = name;
        geomInst->vertexBuffer = vertexBuffer;
        geomInst->triangleBuffer.initialize(optixEnv->cuContext, g_bufferType, triangles, stream);
        geomInst->optixGeomInst = optixEnv->scene.createGeometryInstance();
        geomInst->optixGeomInst.setVertexBuffer(*vertexBuffer);
        geomInst->optixGeomInst.setTriangleBuffer(geomInst->triangleBuffer);
        geomInst->optixGeomInst.setMaterialCount(1, optixu::BufferView());
        geomInst->optixGeomInst.setMaterial(0, 0, optixEnv->material);
        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = vertexBuffer->getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = geomInst->triangleBuffer.getROBuffer<enableBufferOobCheck>();
        geomInst->optixGeomInst.setUserData(geomData);

        fileGroup.geomInsts.push_back(geomInst);
    }

    fileGroup.serialID = optixEnv->geomInstFileGroupSerialID++;
    optixEnv->geomInstFileGroups[fileGroup.serialID] = fileGroup;
}




class GeometryInstanceList {
    struct FileGroupState {
        std::set<uint32_t> selectedIndices;
    };

    const OptiXEnv &m_optixEnv;
    // From GeometryInstanceFileGroup's SerialID
    std::unordered_map<uint32_t, FileGroupState> m_fileGroupStates;

public:
    GeometryInstanceList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv) {}

    void show() {
        if (ImGui::BeginTable("##geomInstList", 4,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                              ImVec2(0, 500))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("#Prims", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);

            ImGui::TableHeadersRow();

            for (const auto &kv : m_optixEnv.geomInstFileGroups) {
                const GeometryInstanceFileGroup &fileGroup = kv.second;
                if (m_fileGroupStates.count(fileGroup.serialID) == 0)
                    m_fileGroupStates[fileGroup.serialID] = FileGroupState();
                FileGroupState &fileGroupState = m_fileGroupStates.at(fileGroup.serialID);

                ImGui::TableNextRow();
                bool allSelected = fileGroupState.selectedIndices.size() == fileGroup.geomInsts.size();
                ImGuiTreeNodeFlags fileFlags = (ImGuiTreeNodeFlags_SpanFullWidth |
                                                ImGuiTreeNodeFlags_OpenOnArrow);
                if (allSelected)
                    fileFlags |= ImGuiTreeNodeFlags_Selected;
                ImGui::TableNextColumn();
                bool open = ImGui::TreeNodeEx(fileGroup.name.c_str(), fileFlags);
                bool onArrow =
                    (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) < ImGui::GetTreeNodeToLabelSpacing();
                if (ImGui::IsItemClicked() && !onArrow) {
                    if (!ImGui::GetIO().KeyCtrl) {
                        // JP: Ctrlを押していない状態でクリックした場合は他のファイルの選択を全て解除する。
                        bool deselectedOthers = false;
                        for (auto &fileGroupKv : m_fileGroupStates) {
                            if (fileGroupKv.first != fileGroup.serialID) {
                                if (fileGroupKv.second.selectedIndices.size()) {
                                    deselectedOthers |= true;
                                    fileGroupKv.second.selectedIndices.clear();
                                }
                            }
                        }
                        // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                        //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                        if (deselectedOthers)
                            allSelected = false;
                    }

                    if (allSelected) {
                        fileGroupState.selectedIndices.clear();
                    }
                    else {
                        for (int i = 0; i < fileGroup.geomInsts.size(); ++i)
                            fileGroupState.selectedIndices.insert(i);
                    }
                }
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("--");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("--");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("--");
                if (open) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.0f, 1.0f, 0.5f, 0.25f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.0f, 1.0f, 0.5f, 0.5f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.0f, 1.0f, 0.5f, 0.75f));
                    for (int i = 0; i < fileGroup.geomInsts.size(); ++i) {
                        const GeometryInstanceRef &geomInst = fileGroup.geomInsts[i];
                        ImGuiTreeNodeFlags instFlags =
                            (ImGuiTreeNodeFlags_Leaf |
                             ImGuiTreeNodeFlags_Bullet |
                             ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_SpanFullWidth);
                        bool geomInstSelected = fileGroupState.selectedIndices.count(i) > 0;
                        if (geomInstSelected)
                            instFlags |= ImGuiTreeNodeFlags_Selected;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TreeNodeEx(geomInst->name.c_str(), instFlags);
                        if (ImGui::IsItemClicked()) {
                            if (!ImGui::GetIO().KeyCtrl) {
                                bool deselectedOthers = false;
                                for (auto &fileGroupKv : m_fileGroupStates) {
                                    if (fileGroupKv.first != fileGroup.serialID) {
                                        if (fileGroupKv.second.selectedIndices.size()) {
                                            deselectedOthers |= true;
                                            fileGroupKv.second.selectedIndices.clear();
                                        }
                                    }
                                    else {
                                        deselectedOthers |= fileGroupKv.second.selectedIndices.size() > 1;
                                        fileGroupKv.second.selectedIndices.clear();
                                        if (geomInstSelected)
                                            fileGroupKv.second.selectedIndices.insert(i);
                                    }
                                }
                                // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                                //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                                if (deselectedOthers)
                                    geomInstSelected = false;
                            }

                            if (geomInstSelected)
                                fileGroupState.selectedIndices.erase(i);
                            else
                                fileGroupState.selectedIndices.insert(i);
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", geomInst->serialID);
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", geomInst->triangleBuffer.numElements());
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", geomInst.use_count() - 1);
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::TreePop();
                }
            }

            ImGui::EndTable();
        }
    }

    uint32_t getNumSelected() const {
        uint32_t sum = 0;
        for (auto it = m_fileGroupStates.cbegin(); it != m_fileGroupStates.cend(); ++it)
            sum += it->second.selectedIndices.size();
        return sum;
    }

    void clearSelection() {
        m_fileGroupStates.clear();
    }

    void loopForSelected(const std::function<bool(uint32_t, const std::set<uint32_t> &)> &func) const {
        for (auto it = m_fileGroupStates.cbegin(); it != m_fileGroupStates.cend(); ++it) {
            bool b = func(it->first, it->second.selectedIndices);
            if (!b)
                break;
        }
    }
};

class GeometryGroupList {
    const OptiXEnv &m_optixEnv;
    std::set<uint32_t> m_selectedGroups;
    uint32_t m_activeGroupSerialID;
    std::set<uint32_t> m_selectedIndicesForActiveGroup;

public:
    GeometryGroupList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv) {}

    void show(bool* selectionChanged) {
        *selectionChanged = false;
        if (ImGui::BeginTable("##GeomGroupList", 4,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                              ImVec2(0, 500))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("#GeomInsts", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);

            ImGui::TableHeadersRow();

            for (const auto &kv : m_optixEnv.geomGroups) {
                const GeometryGroupRef &geomGroup = kv.second;

                ImGui::TableNextRow();
                ImGuiTreeNodeFlags groupFlags =
                    (ImGuiTreeNodeFlags_SpanFullWidth |
                     ImGuiTreeNodeFlags_OpenOnArrow);
                bool geomGroupSelected = m_selectedGroups.count(geomGroup->serialID) > 0;
                if (geomGroupSelected)
                    groupFlags |= ImGuiTreeNodeFlags_Selected;
                ImGui::TableNextColumn();
                bool open = ImGui::TreeNodeEx(geomGroup->name.c_str(), groupFlags);
                bool onArrow = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) < ImGui::GetTreeNodeToLabelSpacing();
                if (ImGui::IsItemClicked() && !onArrow) {
                    m_selectedIndicesForActiveGroup.clear();

                    if (!ImGui::GetIO().KeyCtrl) {
                        // JP: Ctrlを押していない状態でクリックした場合は他のグループの選択を全て解除する。
                        bool deselectedOthers = m_selectedGroups.size() > 1;
                        m_selectedGroups.clear();
                        if (geomGroupSelected)
                            m_selectedGroups.insert(geomGroup->serialID);
                        // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                        //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                        if (deselectedOthers)
                            geomGroupSelected = false;
                    }

                    if (geomGroupSelected)
                        m_selectedGroups.erase(geomGroup->serialID);
                    else
                        m_selectedGroups.insert(geomGroup->serialID);

                    *selectionChanged = true;
                }
                ImGui::TableNextColumn();
                ImGui::Text("%u", geomGroup->serialID);
                ImGui::TableNextColumn();
                ImGui::Text("%u", static_cast<uint32_t>(kv.second->geomInsts.size()));
                ImGui::TableNextColumn();
                ImGui::Text("%u", kv.second.use_count() - 1);
                if (open) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.0f, 1.0f, 0.5f, 0.25f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.0f, 1.0f, 0.5f, 0.5f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.0f, 1.0f, 0.5f, 0.75f));
                    for (int geomInstIdx = 0; geomInstIdx < geomGroup->geomInsts.size(); ++geomInstIdx) {
                        const GeometryInstanceRef &geomInst = geomGroup->geomInsts[geomInstIdx];
                        ImGuiTreeNodeFlags instFlags =
                            (ImGuiTreeNodeFlags_Leaf |
                             ImGuiTreeNodeFlags_Bullet |
                             ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_SpanFullWidth);
                        bool geomInstSelected = geomGroup->serialID == m_activeGroupSerialID &&
                            m_selectedIndicesForActiveGroup.count(geomInstIdx) > 0;
                        if (geomInstSelected)
                            instFlags |= ImGuiTreeNodeFlags_Selected;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TreeNodeEx(geomInst->name.c_str(), instFlags);
                        if (ImGui::IsItemClicked()) {
                            m_selectedGroups.clear();

                            // JP: 複数選択は同じグループ内でのみ機能する。
                            if (geomGroup->serialID != m_activeGroupSerialID) {
                                m_selectedIndicesForActiveGroup.clear();
                                m_activeGroupSerialID = geomGroup->serialID;
                            }

                            if (!ImGui::GetIO().KeyCtrl) {
                                // JP: Ctrlを押していない状態でクリックした場合は他のインスタンスの選択を全て解除する。
                                bool deselectedOthers = m_selectedIndicesForActiveGroup.size() > 1;
                                m_selectedIndicesForActiveGroup.clear();
                                if (geomInstSelected)
                                    m_selectedIndicesForActiveGroup.insert(geomInstIdx);
                                // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                                //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                                if (deselectedOthers)
                                    geomInstSelected = false;
                            }

                            if (geomInstSelected)
                                m_selectedIndicesForActiveGroup.erase(geomInstIdx);
                            else
                                m_selectedIndicesForActiveGroup.insert(geomInstIdx);

                            *selectionChanged = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", geomInst->serialID);
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted("--");
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", geomInst.use_count() - 1);
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::TreePop();
                }
            }
            ImGui::EndTable();
        }
    }

    void clearSelection() {
        m_selectedIndicesForActiveGroup.clear();
        m_selectedGroups.clear();
    }
    uint32_t getNumSelectedGeomGroups() const {
        return static_cast<uint32_t>(m_selectedGroups.size());
    }
    uint32_t getNumSelectedGeomInsts() const {
        return static_cast<uint32_t>(m_selectedIndicesForActiveGroup.size());
    }
    void loopForSelectedGeomGroups(const std::function<bool(const GeometryGroupRef &)> &func) {
        for (const auto &sid : m_selectedGroups) {
            bool b = func(m_optixEnv.geomGroups.at(sid));
            if (!b)
                break;
        }
    }
    GeometryGroupRef getActiveGeometryGroup() const {
        if (m_selectedIndicesForActiveGroup.size() > 0)
            return m_optixEnv.geomGroups.at(m_activeGroupSerialID);
        return nullptr;
    }
    void callForActiveGeomGroup(
        const std::function<void(const GeometryGroupRef &, const std::set<uint32_t> &)> &func) {
        const GeometryGroupRef &activeGeomGroup = m_optixEnv.geomGroups.at(m_activeGroupSerialID);
        func(activeGeomGroup, m_selectedIndicesForActiveGroup);
    }
    uint32_t getFirstSelectedGeomInstIndex() const {
        if (m_selectedIndicesForActiveGroup.size() > 0)
            return *m_selectedIndicesForActiveGroup.cbegin();
        return 0xFFFFFFFF;
    }
};

class InstanceList {
    const OptiXEnv &m_optixEnv;
    std::set<uint32_t> m_selectedItems;

public:
    InstanceList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv) {}

    void show(bool* selectionChanged) {
        if (ImGui::BeginTable("##InstList", 4,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                              ImVec2(0, 300))) {

            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("AS", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);

            ImGui::TableHeadersRow();

            for (const auto &kv : m_optixEnv.insts) {
                const InstanceRef &inst = kv.second;

                ImGui::TableNextRow();

                bool instSelected = m_selectedItems.count(inst->serialID);
                ImGui::TableNextColumn();
                if (ImGui::Selectable(kv.second->name.c_str(), instSelected, ImGuiSelectableFlags_None)) {
                    if (!ImGui::GetIO().KeyCtrl) {
                        // JP: Ctrlを押していない状態でクリックした場合は他のグループの選択を全て解除する。
                        bool deselectedOthers = m_selectedItems.size() > 1;
                        m_selectedItems.clear();
                        if (instSelected)
                            m_selectedItems.insert(inst->serialID);
                        // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                        //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                        if (deselectedOthers)
                            instSelected = false;
                    }

                    if (instSelected)
                        m_selectedItems.erase(inst->serialID);
                    else
                        m_selectedItems.insert(inst->serialID);

                    *selectionChanged = true;
                }

                ImGui::TableNextColumn();
                ImGui::Text("%u", kv.first);

                ImGui::TableNextColumn();
                if (inst->geomGroup)
                    ImGui::Text("%s", inst->geomGroup->name.c_str());
                else if (inst->group)
                    ImGui::Text("%s", inst->group->name.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%u", inst.use_count() - 1);
            }
            ImGui::EndTable();
        }
    }

    uint32_t getNumSelected() const {
        return static_cast<uint32_t>(m_selectedItems.size());
    }
    void loopForSelected(const std::function<bool(const InstanceRef &)> &func) {
        for (const auto &sid : m_selectedItems) {
            bool b = func(m_optixEnv.insts.at(sid));
            if (!b)
                break;
        }
    }

    const InstanceRef &getFirstSelectedItem() const {
        return m_optixEnv.insts.at(*m_selectedItems.cbegin());
    }
    void clearSelection() {
        m_selectedItems.clear();
    }
};

class GroupList {
    const OptiXEnv &m_optixEnv;
    std::set<uint32_t> m_selectedGroups;
    uint32_t m_activeGroupSerialID;
    std::set<uint32_t> m_selectedIndicesForActiveGroup;

public:
    GroupList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv) {}

    void show(bool* selectionChanged = nullptr) {
        *selectionChanged = false;
        if (ImGui::BeginTable("##GroupList", 4,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY,
                              ImVec2(0, 500))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("#Insts", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);

            ImGui::TableHeadersRow();

            for (const auto &kv : m_optixEnv.groups) {
                const GroupRef &group = kv.second;

                ImGui::TableNextRow();
                ImGuiTreeNodeFlags groupFlags = (ImGuiTreeNodeFlags_SpanFullWidth |
                                                 ImGuiTreeNodeFlags_OpenOnArrow);
                bool groupSelected = m_selectedGroups.count(group->serialID) > 0;
                if (groupSelected)
                    groupFlags |= ImGuiTreeNodeFlags_Selected;
                ImGui::TableNextColumn();
                bool open = ImGui::TreeNodeEx(group->name.c_str(), groupFlags);
                bool onArrow = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) < ImGui::GetTreeNodeToLabelSpacing();
                if (ImGui::IsItemClicked() && !onArrow) {
                    m_selectedIndicesForActiveGroup.clear();

                    if (!ImGui::GetIO().KeyCtrl) {
                        // JP: Ctrlを押していない状態でクリックした場合は他のグループの選択を全て解除する。
                        bool deselectedOthers = m_selectedGroups.size() > 1;
                        m_selectedGroups.clear();
                        if (groupSelected)
                            m_selectedGroups.insert(group->serialID);
                        // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                        //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                        if (deselectedOthers)
                            groupSelected = false;
                    }

                    if (groupSelected)
                        m_selectedGroups.erase(group->serialID);
                    else
                        m_selectedGroups.insert(group->serialID);

                    *selectionChanged = true;
                }
                ImGui::TableNextColumn();
                ImGui::Text("%u", group->serialID);
                ImGui::TableNextColumn();
                ImGui::Text("%u", static_cast<uint32_t>(kv.second->insts.size()));
                ImGui::TableNextColumn();
                ImGui::Text("%u", kv.second.use_count() - 1);
                if (open) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.0f, 1.0f, 0.5f, 0.25f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.0f, 1.0f, 0.5f, 0.5f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.0f, 1.0f, 0.5f, 0.75f));
                    for (int instIdx = 0; instIdx < group->insts.size(); ++instIdx) {
                        const InstanceRef &inst = group->insts[instIdx];
                        ImGuiTreeNodeFlags instFlags =
                            (ImGuiTreeNodeFlags_Leaf |
                             ImGuiTreeNodeFlags_Bullet |
                             ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_SpanFullWidth);
                        bool geomInstSelected = group->serialID == m_activeGroupSerialID &&
                            m_selectedIndicesForActiveGroup.count(instIdx) > 0;
                        if (geomInstSelected)
                            instFlags |= ImGuiTreeNodeFlags_Selected;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TreeNodeEx(inst->name.c_str(), instFlags);
                        if (ImGui::IsItemClicked()) {
                            m_selectedGroups.clear();

                            // JP: 複数選択は同じグループ内でのみ機能する。
                            if (group->serialID != m_activeGroupSerialID) {
                                m_selectedIndicesForActiveGroup.clear();
                                m_activeGroupSerialID = group->serialID;
                            }

                            if (!ImGui::GetIO().KeyCtrl) {
                                // JP: Ctrlを押していない状態でクリックした場合は他のインスタンスの選択を全て解除する。
                                bool deselectedOthers = m_selectedIndicesForActiveGroup.size() > 1;
                                m_selectedIndicesForActiveGroup.clear();
                                if (geomInstSelected)
                                    m_selectedIndicesForActiveGroup.insert(instIdx);
                                // JP: 他のファイルの選択状態の解除があった場合はかならず自身を選択状態に持っていく。
                                //     (allSelectedを一旦falseにすることで続くコードで選択状態になる。)
                                if (deselectedOthers)
                                    geomInstSelected = false;
                            }

                            if (geomInstSelected)
                                m_selectedIndicesForActiveGroup.erase(instIdx);
                            else
                                m_selectedIndicesForActiveGroup.insert(instIdx);

                            *selectionChanged = true;
                        }
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", inst->serialID);
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted("--");
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", inst.use_count() - 1);
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::TreePop();
                }
            }
            ImGui::EndTable();
        }
    }

    void clearSelection() {
        m_selectedIndicesForActiveGroup.clear();
        m_selectedGroups.clear();
    }
    uint32_t getNumSelectedGroups() const {
        return static_cast<uint32_t>(m_selectedGroups.size());
    }
    uint32_t getNumSelectedInsts() const {
        return static_cast<uint32_t>(m_selectedIndicesForActiveGroup.size());
    }
    void loopForSelectedGroups(const std::function<bool(const GroupRef &)> &func) {
        for (const auto &sid : m_selectedGroups) {
            bool b = func(m_optixEnv.groups.at(sid));
            if (!b)
                break;
        }
    }
    GroupRef getActiveGroup() const {
        if (m_selectedIndicesForActiveGroup.size() > 0)
            return m_optixEnv.groups.at(m_activeGroupSerialID);
        return nullptr;
    }
    void callForActiveGroup(const std::function<bool(const GroupRef &, const std::set<uint32_t> &)> &func) {
        const GroupRef &activeGroup = m_optixEnv.groups.at(m_activeGroupSerialID);
        func(activeGroup, m_selectedIndicesForActiveGroup);
    }
};



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "scene_edit";



    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUstream stream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&stream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.exceptionFlags = DEBUG_SELECT(
        (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH),
        OPTIX_EXCEPTION_FLAG_NONE);
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr = readBinaryFile(resourceDir / "ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: このグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: This group is for ray-triangle intersection, so we don't need custom intersection program.
    optixu::HitProgramGroup hitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    pipeline.link(1);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setMissRayTypeCount(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());

    // JP: このプログラムは深いトラバーサルグラフを使用するため明示的な設定が必要。
    // EN: This program uses a deep traversal graph depth, therefore requires explicit configuration.
    {
        // No direct callable programs at all.
        uint32_t dcStackSizeFromTrav = 0;
        uint32_t dcStackSizeFromState = 0;

        // Possible Program Paths:
        // RG - CH
        // RG - MS
        uint32_t ccStackSize = rayGenProgram.getStackSize() +
            std::max(hitProgramGroup.getCHStackSize(), missProgram.getStackSize());
        uint32_t maxTraversableGraphDepth = 5;
        pipeline.setStackSize(dcStackSizeFromTrav, dcStackSizeFromState, ccStackSize, maxTraversableGraphDepth);
    }

    OptiXEnv optixEnv;
    optixEnv.cuContext = cuContext;
    optixEnv.context = optixContext;

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixEnv.material = optixContext.createMaterial();
    optixEnv.material.setHitGroup(Shared::RayType_Primary, hitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    constexpr uint32_t MaxNumGeometryInstances = 8192;
    constexpr uint32_t MaxNumGASs = 512;
    
    optixEnv.scene = optixContext.createScene();
    optixEnv.geomInstSerialID = 0;
    optixEnv.geomInstFileGroupSerialID = 0;
    optixEnv.gasSerialID = 0;
    optixEnv.instSerialID = 0;
    optixEnv.iasSerialID = 0;
    optixEnv.asScratchBuffer.initialize(cuContext, g_bufferType, 32 * 1024 * 1024, 1);

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr int32_t initWindowContentWidth = 1280;
    constexpr int32_t initWindowContentHeight = 720;

    Shared::PipelineLaunchParameters plp;
    plp.travHandle = 0;
    plp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = (float)initWindowContentWidth / initWindowContentHeight;

    pipeline.setScene(optixEnv.scene);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - Scene Edit";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = initWindowContentWidth;
    initConfig.windowContentRenderHeight = initWindowContentHeight;
    initConfig.cameraPosition = make_float3(0, 0, 3.2f);
    initConfig.cameraOrientation = qRotateY(pi_v<float>);
    initConfig.cameraMovingSpeed = 0.01f;
    initConfig.cuContext = cuContext;

    uint32_t sbtIndex = -1;
    cudau::Buffer* curHitGroupSBT;
    OptixTraversableHandle curTravHandle = 0;

    GUIFramework framework;
    framework.initialize(initConfig);

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontDefault();

    ImFont* fontForFileDialog = nullptr;
    std::filesystem::path fontPath = getExecutableDirectory() / "fonts/RictyDiminished-Regular-fixed.ttf";
    if (std::filesystem::exists(fontPath)) {
        fontForFileDialog = io.Fonts->AddFontFromFileTTF(fontPath.string().c_str(), 14.0f, nullptr,
                                                         io.Fonts->GetGlyphRangesJapanese());
    }
    else {
        hpprintf("Font for Japanese not found: %s\n", fontPath.u8string().c_str());
    }

    FileDialog fileDialog;
    fileDialog.setFont(fontForFileDialog);
    fileDialog.setFlags(FileDialog::Flag_FileSelection);
    //fileDialog.setFlags(FileDialog::Flag_FileSelection |
    //                    FileDialog::Flag_DirectorySelection |
    //                    FileDialog::Flag_MultipleSelection);

    cudau::Array outputArray;
    outputArray.initializeFromGLTexture2D(
        cuContext, framework.getOutputTexture().getHandle(),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputBufferSurfaceHolder.initialize({ &outputArray });

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;

        // Camera Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&args.cameraPosition));
            static float rollPitchYaw[3];
            args.tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw))
                args.cameraOrientation = qFromEulerAngles(
                    rollPitchYaw[0] * pi_v<float> / 180,
                    rollPitchYaw[1] * pi_v<float> / 180,
                    rollPitchYaw[2] * pi_v<float> / 180);
            ImGui::Text("Pos. Speed (T/G): %g", args.cameraPositionalMovingSpeed);

            ImGui::End();
        }

        plp.camera.position = args.cameraPosition;
        plp.camera.orientation = args.tempCameraOrientation.toMatrix3x3();



        // Scene Window
        static int32_t travIndex = -1;
        static std::vector<std::string> traversableNames;
        static std::vector<OptixTraversableHandle> traversables;
        bool traversablesUpdated = false;
        {
            ImGui::SetNextWindowPos(ImVec2(984, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Scene", nullptr,
                         ImGuiWindowFlags_None);

            if (ImGui::Button("Open"))
                fileDialog.show();
            if (fileDialog.drawAndGetResult() == FileDialog::Result::Result_OK) {
                static std::vector<std::filesystem::directory_entry> entries;
                fileDialog.calcEntries(&entries);
                
                loadFile(entries[0], curStream, &optixEnv);
            }

            if (ImGui::Combo("Target", &travIndex,
                             [](void* data, int idx, const char** out_text) {
                                 if (idx < 0)
                                     return false;
                                 auto nameList = reinterpret_cast<std::string*>(data);
                                 *out_text = nameList[idx].c_str();
                                 return true;
                             }, traversableNames.data(), traversables.size())) {
                curTravHandle = traversables[travIndex];
            }

            if (ImGui::BeginTabBar("Scene", ImGuiTabBarFlags_None)) {
                if (ImGui::BeginTabItem("Geom Inst")) {
                    static GeometryInstanceList geomInstList(optixEnv);
                    geomInstList.show();

                    uint32_t numSelectedGeomInsts = geomInstList.getNumSelected();
                    bool geomInstsSelected = numSelectedGeomInsts > 0;
                    bool allUnused = numSelectedGeomInsts > 0;
                    geomInstList.loopForSelected(
                        [&optixEnv, &allUnused]
                    (uint32_t geomInstFileGroupSerialID, const std::set<uint32_t> &indices) {
                            const GeometryInstanceFileGroup &fileGroup = optixEnv.geomInstFileGroups.at(geomInstFileGroupSerialID);
                            for (uint32_t index : indices) {
                                allUnused &= fileGroup.geomInsts[index].use_count() == 1;
                                if (!allUnused)
                                    break;
                            }
                            return allUnused;
                        });
                    bool selectedGeomInstsRemovable = allUnused && numSelectedGeomInsts > 0;

                    if (ImGui::Button("Create a GAS", geomInstsSelected)) {
                        uint32_t serialID = optixEnv.gasSerialID++;
                        GeometryGroupRef geomGroup =
                            make_shared_with_deleter<GeometryGroup>(GeometryGroup::finalize);
                        char name[256];
                        sprintf_s(name, "GAS-%u", serialID);
                        geomGroup->optixEnv = &optixEnv;
                        geomGroup->serialID = serialID;
                        geomGroup->name = name;
                        geomGroup->optixGAS = optixEnv.scene.createGeometryAccelerationStructure();
                        geomGroup->optixGAS.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
                        geomGroup->optixGas.setMaterialSetCount(1);
                        geomGroup->optixGAS.setRayTypeCount(0, Shared::NumRayTypes);
                        geomGroup->preTransforms.resize(numSelectedGeomInsts);
                        geomGroup->preTransformBuffer.initialize(
                            optixEnv.cuContext, g_bufferType, numSelectedGeomInsts);

                        std::array<float, 12>* preTransforms = geomGroup->preTransformBuffer.map(curStream);
                        geomInstList.loopForSelected(
                            [&optixEnv, &geomGroup, &preTransforms, &curStream]
                        (uint32_t geomInstFileGroupSerialID, const std::set<uint32_t> &indices) {
                                const GeometryInstanceFileGroup &fileGroup = optixEnv.geomInstFileGroups.at(geomInstFileGroupSerialID);
                                for (uint32_t index : indices) {
                                    const GeometryInstanceRef &geomInst = fileGroup.geomInsts[index];
                                    geomGroup->geomInsts.push_back(geomInst);
                                    uint32_t indexInGAS = geomGroup->geomInsts.size() - 1;

                                    GeometryInstancePreTransform &preTransform =
                                        geomGroup->preTransforms[indexInGAS];
                                    preTransform.scale = float3(1.0f, 1.0f, 1.0f);
                                    preTransform.rollPitchYaw[0] = 0.0f;
                                    preTransform.rollPitchYaw[1] = 0.0f;
                                    preTransform.rollPitchYaw[2] = 0.0f;
                                    preTransform.position = float3(0.0f, 0.0f, 0.0f);

                                    Shared::GASChildData gasChildData;
                                    gasChildData.setPreTransform(
                                        preTransform.scale,
                                        preTransform.rollPitchYaw,
                                        preTransform.position);

                                    float* raw = preTransforms[indexInGAS].data();
                                    Matrix3x3 matSR =
                                        gasChildData.orientation.toMatrix3x3()
                                        * scale3x3(gasChildData.scale);
                                    raw[0] = matSR.m00; raw[1] = matSR.m01; raw[2] = matSR.m02; raw[3] = gasChildData.translation.x;
                                    raw[4] = matSR.m10; raw[5] = matSR.m11; raw[6] = matSR.m12; raw[7] = gasChildData.translation.y;
                                    raw[8] = matSR.m20; raw[9] = matSR.m21; raw[10] = matSR.m22; raw[11] = gasChildData.translation.z;
                                    CUdeviceptr preTransformPtr =
                                        geomGroup->preTransformBuffer.getCUdeviceptrAt(indexInGAS);

                                    geomGroup->optixGAS.addChild(
                                        geomInst->optixGeomInst, preTransformPtr, gasChildData);
                                }
                                return true;
                            });
                        geomGroup->preTransformBuffer.unmap(curStream);

                        traversablesUpdated = true;

                        optixEnv.geomGroups[serialID] = geomGroup;
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove", selectedGeomInstsRemovable)) {
                        geomInstList.loopForSelected(
                            [&optixEnv]
                        (uint32_t geomInstFileGroupSerialID, const std::set<uint32_t> &indices) {
                                GeometryInstanceFileGroup &fileGroup =
                                    optixEnv.geomInstFileGroups.at(geomInstFileGroupSerialID);
                                // JP: serialIDに基づいているためまず安全。
                                for (auto it = indices.crbegin(); it != indices.crend(); ++it)
                                    fileGroup.geomInsts.erase(fileGroup.geomInsts.cbegin() + *it);
                                if (fileGroup.geomInsts.size() == 0)
                                    optixEnv.geomInstFileGroups.erase(geomInstFileGroupSerialID);
                                return true;
                            });
                        geomInstList.clearSelection();
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Clear Selection"))
                        geomInstList.clearSelection();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Geom Group")) {
                    static GeometryGroupList geomGroupList(optixEnv);

                    bool selectionChanged;
                    geomGroupList.show(&selectionChanged);

                    static bool allGroupUnused = true;
                    static bool geomGroupSelected = false;
                    static bool geomInstsSelected = false;
                    static bool singleGeomInstSelected = false;

                    const auto onSelectionChange = [&]() {
                        allGroupUnused = true;
                        geomGroupList.loopForSelectedGeomGroups(
                            [&optixEnv](const GeometryGroupRef &geomGroup) {
                                allGroupUnused &= geomGroup.use_count() == 1;
                                return allGroupUnused;
                            });
                        geomGroupSelected = geomGroupList.getNumSelectedGeomGroups() > 0;
                        uint32_t numGeomInstsSelected = geomGroupList.getNumSelectedGeomInsts();
                        geomInstsSelected = numGeomInstsSelected > 0;
                        singleGeomInstSelected = numGeomInstsSelected == 1;
                    };

                    if (selectionChanged)
                        onSelectionChange();

                    if (ImGui::Button("Create Instances", geomGroupSelected)) {
                        geomGroupList.loopForSelectedGeomGroups(
                            [&optixEnv, &curStream](const GeometryGroupRef &geomGroup) {
                                uint32_t serialID = optixEnv.instSerialID++;
                                InstanceRef inst = make_shared_with_deleter<Instance>(Instance::finalize);
                                inst->optixEnv = &optixEnv;
                                char name[256];
                                sprintf_s(name, "Instance-%u", serialID);
                                inst->serialID = serialID;
                                inst->name = name;
                                inst->scale = float3(1.0f, 1.0f, 1.0f);
                                inst->rollPitchYaw[0] = 0.0f;
                                inst->rollPitchYaw[1] = 0.0f;
                                inst->rollPitchYaw[2] = 0.0f;
                                inst->position = float3(0.0f, 0.0f, 0.0f);

                                Matrix3x3 srMat =
                                    qFromEulerAngles(inst->rollPitchYaw[0],
                                                     inst->rollPitchYaw[1],
                                                     inst->rollPitchYaw[2]).toMatrix3x3() *
                                    scale3x3(inst->scale);

                                inst->geomGroup = geomGroup;
                                geomGroup->parentInsts.insert(inst);
                                inst->optixInst = optixEnv.scene.createInstance();
                                inst->optixInst.setChild(geomGroup->optixGAS);
                                float tr[] = {
                                    srMat.m00, srMat.m01, srMat.m02, inst->position.x,
                                    srMat.m10, srMat.m11, srMat.m12, inst->position.y,
                                    srMat.m20, srMat.m21, srMat.m22, inst->position.z,
                                };
                                inst->optixInst.setTransform(tr);

                                optixEnv.insts[serialID] = inst;

                                return true;
                            });
                    }

                    ImGui::SameLine();
                    bool removeIsActive = (allGroupUnused && geomGroupSelected) || geomInstsSelected;
                    if (ImGui::Button("Remove", removeIsActive)) {
                        if (geomGroupSelected) {
                            geomGroupList.loopForSelectedGeomGroups(
                                [&optixEnv](const GeometryGroupRef &geomGroup) {
                                    optixEnv.geomGroups.erase(geomGroup->serialID);
                                    return true;
                                });
                        }
                        else if (geomInstsSelected) {
                            const GeometryGroupRef &geomGroup = geomGroupList.getActiveGeometryGroup();
                            geomGroupList.callForActiveGeomGroup(
                                [&optixEnv]
                            (const GeometryGroupRef &geomGroup, const std::set<uint32_t> &selectedIndices) {
                                    // JP: serialIDに基づいているためまず安全。
                                    for (auto it = selectedIndices.crbegin(); it != selectedIndices.crend(); ++it) {
                                        uint32_t geomInstIdx = *it;
                                        const GeometryInstanceRef &geomInst = geomGroup->geomInsts[geomInstIdx];
                                        geomGroup->optixGAS.removeChildAt(geomInstIdx);
                                        geomGroup->geomInsts.erase(geomGroup->geomInsts.cbegin() + geomInstIdx);
                                        geomGroup->preTransforms.erase(geomGroup->preTransforms.cbegin() + geomInstIdx);
                                    }
                                });
                            std::array<float, 12>* preTransforms = geomGroup->preTransformBuffer.map(curStream);
                            for (int i = 0; i < geomGroup->geomInsts.size(); ++i) {
                                Shared::GASChildData gasChildData;
                                geomGroup->optixGAS.getChildUserData(i, &gasChildData);

                                float* raw = preTransforms[i].data();
                                Matrix3x3 matSR = gasChildData.orientation.toMatrix3x3() * scale3x3(gasChildData.scale);
                                raw[0] = matSR.m00; raw[1] = matSR.m01; raw[2] = matSR.m02; raw[3] = gasChildData.translation.x;
                                raw[4] = matSR.m10; raw[5] = matSR.m11; raw[6] = matSR.m12; raw[7] = gasChildData.translation.y;
                                raw[8] = matSR.m20; raw[9] = matSR.m21; raw[10] = matSR.m22; raw[11] = gasChildData.translation.z;
                            }
                            geomGroup->preTransformBuffer.unmap(curStream);
                            geomGroup->propagateMarkDirty();
                        }
                        geomGroupList.clearSelection();
                        onSelectionChange();

                        traversablesUpdated = true;
                    }

                    if (ImGui::Button("Clear Selection"))
                        geomGroupList.clearSelection();



                    // TODO: Multiple Selection/Editing
                    ImGui::Separator();
                    if (!singleGeomInstSelected)
                        ImGui::PushDisabledStyle();

                    GeometryGroupRef activeGroup = geomGroupList.getActiveGeometryGroup();
                    uint32_t selectedGeomInstIndex = geomGroupList.getFirstSelectedGeomInstIndex();
                    GeometryInstanceRef selectedGeomInst = selectedGeomInstIndex != 0xFFFFFFFF ?
                        activeGroup->geomInsts[selectedGeomInstIndex] : nullptr;
                    GeometryInstancePreTransform dummyPreTransform;
                    GeometryInstancePreTransform &preTransform = selectedGeomInstIndex != 0xFFFFFFFF ?
                        activeGroup->preTransforms[selectedGeomInstIndex] : dummyPreTransform;

                    ImGui::Text("%s", singleGeomInstSelected ? selectedGeomInst->name.c_str() : "Not selected");
                    static float instScale[3];
                    static float instOrientation[3];
                    static float instPosition[3];
                    bool srtUpdated = false;
                    if (singleGeomInstSelected) {
                        std::copy_n(reinterpret_cast<float*>(&preTransform.scale), 3, instScale);
                        std::copy_n(preTransform.rollPitchYaw, 3, instOrientation);
                        instOrientation[0] *= 180 / pi_v<float>;
                        instOrientation[1] *= 180 / pi_v<float>;
                        instOrientation[2] *= 180 / pi_v<float>;
                        std::copy_n(reinterpret_cast<float*>(&preTransform.position), 3, instPosition);
                    }
                    else {
                        instScale[0] = instScale[1] = instScale[2] = 0.0f;
                        instOrientation[0] = instOrientation[1] = instOrientation[2] = 0.0f;
                        instPosition[0] = instPosition[1] = instPosition[2] = 0.0f;
                    }
                    srtUpdated |= ImGui::InputFloat3("Scale", instScale, "%.5f");
                    srtUpdated |= ImGui::InputFloat3("Roll/Pitch/Yaw", instOrientation, "%.5f");
                    srtUpdated |= ImGui::InputFloat3("Position", instPosition, "%.5f");
                    if (singleGeomInstSelected && srtUpdated) {
                        preTransform.scale = float3(instScale[0], instScale[1], instScale[2]);
                        preTransform.rollPitchYaw[0] = instOrientation[0] * pi_v<float> / 180;
                        preTransform.rollPitchYaw[1] = instOrientation[1] * pi_v<float> / 180;
                        preTransform.rollPitchYaw[2] = instOrientation[2] * pi_v<float> / 180;
                        preTransform.position = float3(instPosition[0], instPosition[1], instPosition[2]);

                        Shared::GASChildData gasChildData = {};
                        gasChildData.setPreTransform(
                            preTransform.scale,
                            preTransform.rollPitchYaw,
                            preTransform.position);
                        activeGroup->optixGAS.setChildUserData(selectedGeomInstIndex, gasChildData);

                        std::array<float, 12> dstPreTransform;
                        float* raw = dstPreTransform.data();
                        Matrix3x3 matSR = gasChildData.orientation.toMatrix3x3() * scale3x3(gasChildData.scale);
                        raw[0] = matSR.m00; raw[1] = matSR.m01; raw[2] = matSR.m02; raw[3] = gasChildData.translation.x;
                        raw[4] = matSR.m10; raw[5] = matSR.m11; raw[6] = matSR.m12; raw[7] = gasChildData.translation.y;
                        raw[8] = matSR.m20; raw[9] = matSR.m21; raw[10] = matSR.m22; raw[11] = gasChildData.translation.z;
                        CUDADRV_CHECK(cuMemcpyHtoDAsync(
                            activeGroup->preTransformBuffer.getCUdeviceptrAt(selectedGeomInstIndex),
                            &dstPreTransform,
                            sizeof(dstPreTransform),
                            curStream));

                        activeGroup->optixGAS.markDirty();
                        activeGroup->propagateMarkDirty();

                        // TODO: GeomInstの変換の更新はSBTレイアウトを無効化する必要がないが、
                        //       現状GASのmarkDirty()が必ずSBTレイアウトの無効化も実行するようになっているため
                        //       ここではSBTレイアウトの更新が必要であるとする。
                        //       => ここではリフィッティングの代わりとしてのリビルドになるのでmarkDirty()を
                        //          そもそも呼ばなくてよいのでは？
                        //          ただしこのサンプルではリフィッティングを使っていないので上位階層にはdirty()を
                        //          伝える必要がある。
                        traversablesUpdated = true;
                    }

                    if (!singleGeomInstSelected)
                        ImGui::PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Inst")) {
                    static InstanceList instList(optixEnv);

                    bool selectionChanged;
                    instList.show(&selectionChanged);

                    static bool instsSelected = false;
                    static bool singleInstSelected = false;
                    static bool allUnused = true;
                    static bool selectedInstsRemovable = false;

                    const auto onSelectionChange = [&]() {
                        instsSelected = instList.getNumSelected() > 0;
                        singleInstSelected = instList.getNumSelected() == 1;
                        allUnused = true;
                        instList.loopForSelected(
                            [](const InstanceRef &inst) {
                                allUnused &= inst.use_count() == 1;
                                return allUnused;
                            });
                        selectedInstsRemovable = allUnused && instList.getNumSelected() > 0;
                    };

                    if (selectionChanged)
                        onSelectionChange();

                    if (ImGui::Button("Create an IAS", instsSelected)) {
                        uint32_t serialID = optixEnv.iasSerialID++;
                        GroupRef group = make_shared_with_deleter<Group>(Group::finalize);
                        group->optixEnv = &optixEnv;
                        char name[256];
                        sprintf_s(name, "IAS-%u", serialID);
                        group->serialID = serialID;
                        group->name = name;
                        group->optixIAS = optixEnv.scene.createInstanceAccelerationStructure();
                        group->optixIAS.setConfiguration(optixu::ASTradeoff::PreferFastBuild);

                        instList.loopForSelected(
                            [&group](const InstanceRef &inst) {
                                group->insts.push_back(inst);
                                group->optixIAS.addChild(inst->optixInst);
                                inst->parentGroups.insert(group);
                                return true;
                            });

                        optixEnv.groups[serialID] = group;
                        traversablesUpdated = true;
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove", selectedInstsRemovable)) {
                        instList.loopForSelected(
                            [&optixEnv](const InstanceRef &inst) {
                                InstanceWRef instWRef = std::weak_ptr(inst);
                                if (inst->geomGroup)
                                    inst->geomGroup->parentInsts.erase(instWRef);
                                else if (inst->group)
                                    inst->group->parentInsts.erase(instWRef);
                                optixEnv.insts.erase(inst->serialID);
                                return true;
                            });
                        instList.clearSelection();
                        onSelectionChange();
                    }



                    // TODO: Multiple Selection/Editing
                    ImGui::Separator();
                    if (!singleInstSelected)
                        ImGui::PushDisabledStyle();

                    const InstanceRef &selectedInst = singleInstSelected ?
                        instList.getFirstSelectedItem() : InstanceRef();

                    ImGui::Text("%s", singleInstSelected ? selectedInst->name.c_str() : "Not selected");
                    static float instScale[3];
                    static float instOrientation[3];
                    static float instPosition[3];
                    bool srtUpdated = false;
                    if (singleInstSelected) {
                        std::copy_n(reinterpret_cast<float*>(&selectedInst->scale), 3, instScale);
                        std::copy_n(selectedInst->rollPitchYaw, 3, instOrientation);
                        instOrientation[0] *= 180 / pi_v<float>;
                        instOrientation[1] *= 180 / pi_v<float>;
                        instOrientation[2] *= 180 / pi_v<float>;
                        std::copy_n(reinterpret_cast<float*>(&selectedInst->position), 3, instPosition);
                    }
                    else {
                        instScale[0] = instScale[1] = instScale[2] = 0.0f;
                        instOrientation[0] = instOrientation[1] = instOrientation[2] = 0.0f;
                        instPosition[0] = instPosition[1] = instPosition[2] = 0.0f;
                    }
                    srtUpdated |= ImGui::InputFloat3("Scale", instScale, "%.5f");
                    srtUpdated |= ImGui::InputFloat3("Roll/Pitch/Yaw", instOrientation, "%.5f");
                    srtUpdated |= ImGui::InputFloat3("Position", instPosition, "%.5f");
                    if (singleInstSelected && srtUpdated) {
                        selectedInst->scale = float3(instScale[0], instScale[1], instScale[2]);
                        selectedInst->rollPitchYaw[0] = instOrientation[0] * pi_v<float> / 180;
                        selectedInst->rollPitchYaw[1] = instOrientation[1] * pi_v<float> / 180;
                        selectedInst->rollPitchYaw[2] = instOrientation[2] * pi_v<float> / 180;
                        selectedInst->position = float3(instPosition[0], instPosition[1], instPosition[2]);

                        Matrix3x3 srMat =
                            qFromEulerAngles(selectedInst->rollPitchYaw[0],
                                             selectedInst->rollPitchYaw[1],
                                             selectedInst->rollPitchYaw[2]).toMatrix3x3() *
                            scale3x3(selectedInst->scale);

                        float tr[] = {
                            srMat.m00, srMat.m01, srMat.m02, selectedInst->position.x,
                            srMat.m10, srMat.m11, srMat.m12, selectedInst->position.y,
                            srMat.m20, srMat.m21, srMat.m22, selectedInst->position.z,
                        };
                        selectedInst->optixInst.setTransform(tr);

                        selectedInst->propagateMarkDirty();
                    }

                    if (!singleInstSelected)
                        ImGui::PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Group")) {
                    static GroupList groupList(optixEnv);

                    bool selectionChanged;
                    groupList.show(&selectionChanged);

                    static bool allGroupUnused = true;
                    static uint32_t numGroupsSelected = 0;
                    static uint32_t numInstsSelected = 0;
                    static bool removeIsActive = false;

                    const auto onSelectionChange = [&]() {
                        bool allGroupUnused = true;
                        groupList.loopForSelectedGroups(
                            [&optixEnv, &allGroupUnused](const GroupRef &group) {
                                allGroupUnused &= group.use_count() == 1;
                                return allGroupUnused;
                            });
                        numGroupsSelected = groupList.getNumSelectedGroups();
                        numInstsSelected = groupList.getNumSelectedInsts();
                        removeIsActive = (allGroupUnused && numGroupsSelected > 0) || numInstsSelected > 0;
                    };

                    if (selectionChanged)
                        onSelectionChange();

                    if (ImGui::Button("Create Instances", numGroupsSelected > 0)) {
                        groupList.loopForSelectedGroups(
                            [&optixEnv, &curStream](const GroupRef &group) {
                                uint32_t serialID = optixEnv.instSerialID++;
                                InstanceRef inst = make_shared_with_deleter<Instance>(Instance::finalize);
                                inst->optixEnv = &optixEnv;
                                char name[256];
                                sprintf_s(name, "Instance-%u", serialID);
                                inst->serialID = serialID;
                                inst->name = name;
                                inst->scale = float3(1.0f, 1.0f, 1.0f);
                                inst->rollPitchYaw[0] = 0.0f;
                                inst->rollPitchYaw[1] = 0.0f;
                                inst->rollPitchYaw[2] = 0.0f;
                                inst->position = float3(0.0f, 0.0f, 0.0f);

                                Matrix3x3 srMat =
                                    qFromEulerAngles(inst->rollPitchYaw[0],
                                                     inst->rollPitchYaw[1],
                                                     inst->rollPitchYaw[2]).toMatrix3x3() *
                                    scale3x3(inst->scale);

                                inst->group = group;
                                group->parentInsts.insert(inst);
                                inst->optixInst = optixEnv.scene.createInstance();
                                inst->optixInst.setChild(group->optixIAS);
                                float tr[] = {
                                    srMat.m00, srMat.m01, srMat.m02, inst->position.x,
                                    srMat.m10, srMat.m11, srMat.m12, inst->position.y,
                                    srMat.m20, srMat.m21, srMat.m22, inst->position.z,
                                };
                                inst->optixInst.setTransform(tr);

                                optixEnv.insts[serialID] = inst;

                                return true;
                            });
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove", removeIsActive)) {
                        if (numGroupsSelected > 0) {
                            groupList.loopForSelectedGroups(
                                [&optixEnv](const GroupRef &group) {
                                    optixEnv.groups.erase(group->serialID);
                                    return true;
                                });
                        }
                        else if (numInstsSelected > 0) {
                            groupList.callForActiveGroup(
                                [&optixEnv](const GroupRef &group, const std::set<uint32_t> &selectedIndices) {
                                    // JP: serialIDに基づいているためまず安全。
                                    for (auto it = selectedIndices.crbegin(); it != selectedIndices.crend(); ++it) {
                                        uint32_t instIdx = *it;
                                        group->optixIAS.removeChildAt(instIdx);
                                        group->insts.erase(group->insts.cbegin() + instIdx);
                                        group->propagateMarkDirty();
                                    }
                                    return true;
                                });
                        }
                        traversablesUpdated = true;
                        groupList.clearSelection();
                        onSelectionChange();
                    }

                    if (ImGui::Button("Clear Selection"))
                        groupList.clearSelection();

                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::End();
        }



        for (const auto &kv : optixEnv.geomGroups) {
            const GeometryGroupRef &geomGroup = kv.second;
            if (geomGroup->optixGAS.isReady())
                continue;

            OptixAccelBufferSizes bufferSizes;
            geomGroup->optixGAS.prepareForBuild(&bufferSizes);
            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1, curStream);
            hpprintf("GAS: %s\n", kv.second->name.c_str());
            hpprintf("AS Size: %llu bytes\n", bufferSizes.outputSizeInBytes);
            hpprintf("Scratch Size: %llu bytes\n", bufferSizes.tempSizeInBytes);
            // JP: ASのメモリをGPUが使用中に確保しなおすのは危険なため使用の完了を待つ。
            //     CPU/GPUの非同期実行を邪魔したくない場合、完全に別のバッファーを用意してそれと切り替える必要がある。
            // EN: It is dangerous to reallocate AS memory during the GPU is using it,
            //     so wait the completion of use.
            //     You need to prepare a completely separated buffer and switch the current with it
            //     if you don't want to interfere CPU/GPU asynchronous execution.
            if (geomGroup->optixGasMem.isInitialized()) {
                if (bufferSizes.outputSizeInBytes > geomGroup->optixGasMem.sizeInBytes()) {
                    CUDADRV_CHECK(cuStreamSynchronize(curStream));
                    geomGroup->optixGasMem.resize(bufferSizes.outputSizeInBytes, 1, curStream);
                    // TODO: curStreamを待つのではなくresize()にdefault streamを渡して待つようにしても良いかもしれない。
                }
            }
            else {
                CUDADRV_CHECK(cuStreamSynchronize(curStream));
                geomGroup->optixGasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
            }
            geomGroup->optixGAS.rebuild(curStream, geomGroup->optixGasMem, optixEnv.asScratchBuffer);
        }

        if (!optixEnv.scene.shaderBindingTableLayoutIsReady()) {
            sbtIndex = (sbtIndex + 1) % 2;
            curHitGroupSBT = &optixEnv.hitGroupSBT[sbtIndex];

            size_t hitGroupSbtSize;
            optixEnv.scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
            if (curHitGroupSBT->isInitialized()) {
                curHitGroupSBT->resize(hitGroupSbtSize, 1, curStream);
            }
            else {
                curHitGroupSBT->initialize(cuContext, g_bufferType, hitGroupSbtSize, 1);
                curHitGroupSBT->setMappedMemoryPersistent(true);
            }
            pipeline.setHitGroupShaderBindingTable(*curHitGroupSBT, curHitGroupSBT->getMappedPointer());
        }

        for (const auto &kv : optixEnv.groups) {
            const GroupRef &group = kv.second;
            if (group->optixIAS.isReady())
                continue;

            OptixAccelBufferSizes bufferSizes;
            group->optixIAS.prepareForBuild(&bufferSizes);
            hpprintf("IAS: %s\n", kv.second->name.c_str());
            hpprintf("AS Size: %llu bytes\n", bufferSizes.outputSizeInBytes);
            hpprintf("Scratch Size: %llu bytes\n", bufferSizes.tempSizeInBytes);
            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1, curStream);
            // JP: ASのメモリをGPUが使用中に確保しなおすのは危険なため使用の完了を待つ。
            //     CPU/GPUの非同期実行を邪魔しないためには、完全に別のバッファーを用意してそれと切り替える必要がある。
            // EN: It is dangerous to reallocate AS memory during the GPU is using it,
            //     so wait the completion of use.
            //     You need to prepare a completely separated buffer and switch the current with it
            //     if you don't want to interfere CPU/GPU asynchronous execution.
            if (group->optixIasMem.isInitialized()) {
                if (bufferSizes.outputSizeInBytes > group->optixIasMem.sizeInBytes() ||
                    group->optixIAS.getChildCount() > group->optixInstanceBuffer.numElements()) {
                    CUDADRV_CHECK(cuStreamSynchronize(curStream));
                    group->optixIasMem.resize(bufferSizes.outputSizeInBytes, 1, curStream);
                    group->optixInstanceBuffer.resize(group->optixIAS.getChildCount());
                    // TODO: curStreamを待つのではなくresize()にdefault streamを渡して待つようにしても良いかもしれない。
                }
            }
            else {
                CUDADRV_CHECK(cuStreamSynchronize(curStream));
                group->optixIasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
                group->optixInstanceBuffer.initialize(optixEnv.cuContext, g_bufferType, group->optixIAS.getChildCount());
            }
            group->optixIAS.rebuild(curStream, group->optixInstanceBuffer, group->optixIasMem, optixEnv.asScratchBuffer);
        }

        if (traversablesUpdated) {
            traversables.clear();
            traversableNames.clear();
            for (const auto &kv : optixEnv.groups) {
                const GroupRef &group = kv.second;
                traversables.push_back(group->optixIAS.getHandle());
                traversableNames.push_back(group->name);
            }
            //for (const auto &kv : optixEnv.geomGroups) {
            //    const GeometryGroupRef &group = kv.second;
            //    traversables.push_back(group->optixGAS.getHandle());
            //    traversableNames.push_back(group->name);
            //}

            travIndex = -1;
            for (int i = 0; i < traversables.size(); ++i) {
                if (traversables[i] == curTravHandle) {
                    travIndex = i;
                    break;
                }
            }
            if (travIndex == -1)
                curTravHandle = 0;
        }
        
        // Render
        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        plp.travHandle = curTravHandle;
        plp.resultBuffer = outputBufferSurfaceHolder.getNext();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
        pipeline.launch(curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = false;
        ret.finish = false;

        return ret;
    };

    const auto onResolutionChange = [&]
    (CUstream curStream, uint64_t frameIndex,
     int32_t windowContentWidth, int32_t windowContentHeight) {
         outputArray.finalize();
         outputArray.initializeFromGLTexture2D(
             cuContext, framework.getOutputTexture().getHandle(),
             cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

         // EN: update the pipeline parameters.
         plp.imageSize = int2(windowContentWidth, windowContentHeight);
         plp.camera.aspect = (float)windowContentWidth / windowContentHeight;
    };

    framework.run(onRenderLoop, onResolutionChange);

    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();

    framework.finalize();

    // END: Display the window.
    // ----------------------------------------------------------------



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    optixEnv.groups.clear();
    optixEnv.insts.clear();
    optixEnv.geomGroups.clear();
    optixEnv.geomInstFileGroups.clear();

    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();

    optixEnv.asScratchBuffer.finalize();
    optixEnv.hitGroupSBT[1].finalize();
    optixEnv.hitGroupSBT[0].finalize();
    optixEnv.scene.destroy();

    optixEnv.material.destroy();



    shaderBindingTable.finalize();

    hitProgramGroup.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(stream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
