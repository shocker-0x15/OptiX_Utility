#include "scene_edit_shared.h"

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../common/imgui_file_dialog.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb_image.h"
#include "../common/dds_loader.h"



struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

KeyState g_keyForward;
KeyState g_keyBackward;
KeyState g_keyLeftward;
KeyState g_keyRightward;
KeyState g_keyUpward;
KeyState g_keyDownward;
KeyState g_keyTiltLeft;
KeyState g_keyTiltRight;
KeyState g_keyFasterPosMovSpeed;
KeyState g_keySlowerPosMovSpeed;
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;

float g_cameraPositionalMovingSpeed;
float g_cameraDirectionalMovingSpeed;
float g_cameraTiltSpeed;
Quaternion g_cameraOrientation;
Quaternion g_tempCameraOrientation;
float3 g_cameraPosition;



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
    uint32_t geomInstIndex;
    uint32_t serialID;
    std::string name;
    optixu::GeometryInstance optixGeomInst;
    VertexBufferRef vertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
    bool dataTransfered = false;

    GeometryInstance() : geomInstIndex(SlotFinder::InvalidSlotIndex) {}
    static void finalize(GeometryInstance* p);
};

struct GeometryGroup {
    OptiXEnv* optixEnv;
    uint32_t gasIndex;
    uint32_t serialID;
    std::string name;
    optixu::GeometryAccelerationStructure optixGAS;
    std::vector<GeometryInstanceRef> geomInsts;
    std::vector<Shared::GeometryInstancePreTransform> preTransforms;
    cudau::TypedBuffer<Shared::GeometryInstancePreTransform> preTransformBuffer;
    std::set<InstanceWRef, std::owner_less<InstanceWRef>> parentInsts;
    cudau::Buffer optixGasMem;
    bool dataTransfered = false;

    GeometryGroup() : gasIndex(SlotFinder::InvalidSlotIndex) {}
    static void finalize(GeometryGroup* p);
};

struct Instance {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::Instance optixInst;
    GeometryGroupRef geomGroup;
    std::set<GroupWRef, std::owner_less<GroupWRef>> parentGroups;
    float3 scale;
    float rollPitchYaw[3];
    float3 position;

    static void finalize(Instance* p);
};

struct Group {
    OptiXEnv* optixEnv;
    uint32_t serialID;
    std::string name;
    optixu::InstanceAccelerationStructure optixIAS;
    std::vector<InstanceRef> insts;
    cudau::Buffer optixIasMem;
    cudau::TypedBuffer<OptixInstance> optixInstanceBuffer;

    static void finalize(Group* p);
};

struct OptiXEnv {
    CUcontext cuContext;
    optixu::Context context;
    optixu::Material material;
    optixu::Scene scene;
    cudau::TypedBuffer<Shared::GeometryData> geometryDataBuffer;
    SlotFinder geometryInstSlotFinder;
    cudau::TypedBuffer<Shared::GASData> gasDataBuffer;
    SlotFinder gasSlotFinder;

    uint32_t geomInstSerialID;
    uint32_t gasSerialID;
    uint32_t instSerialID;
    uint32_t iasSerialID;
    std::map<uint32_t, GeometryInstanceRef> geomInsts;
    std::map<uint32_t, GeometryGroupRef> geomGroups;
    std::map<uint32_t, InstanceRef> insts;
    std::map<uint32_t, GroupRef> groups;

    cudau::Buffer asScratchBuffer;

    cudau::Buffer shaderBindingTable[2]; // double buffering
};

void GeometryInstance::finalize(GeometryInstance* p) {
    if (p->geomInstIndex != SlotFinder::InvalidSlotIndex) {
        p->optixGeomInst.destroy();
        p->triangleBuffer.finalize();
        p->optixEnv->geometryInstSlotFinder.setNotInUse(p->geomInstIndex);
    }
    delete p;
}
void GeometryGroup::finalize(GeometryGroup* p) {
    if (p->gasIndex != SlotFinder::InvalidSlotIndex) {
        p->optixGasMem.finalize();
        p->preTransformBuffer.finalize();
        p->optixGAS.destroy();
        p->optixEnv->gasSlotFinder.setNotInUse(p->gasIndex);
    }
    delete p;
}
void Instance::finalize(Instance* p) {
    p->optixInst.destroy();
    delete p;
}
void Group::finalize(Group* p) {
    p->optixInstanceBuffer.finalize();
    p->optixIasMem.finalize();
    p->optixIAS.destroy();
    delete p;
}



void loadFile(const std::filesystem::path &filepath, OptiXEnv* optixEnv) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filepath.string(),
                                             aiProcess_Triangulate |
                                             aiProcess_GenNormals | aiProcess_GenSmoothNormals |
                                             aiProcess_PreTransformVertices);
    if (!scene) {
        hpprintf("Failed to load %s.\n", filepath.c_str());
        return;
    }

    std::string basename = filepath.stem().string();

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
        vertexBuffer->initialize(optixEnv->cuContext, g_bufferType, vertices);

        char name[256];
        sprintf_s(name, "%s-%d", basename.c_str(), meshIdx);
        GeometryInstanceRef geomInst = make_shared_with_deleter<GeometryInstance>(GeometryInstance::finalize);
        uint32_t geomInstIndex = optixEnv->geometryInstSlotFinder.getFirstAvailableSlot();
        optixEnv->geometryInstSlotFinder.setInUse(geomInstIndex);
        geomInst->optixEnv = optixEnv;
        geomInst->geomInstIndex = geomInstIndex;
        geomInst->serialID = optixEnv->geomInstSerialID++;
        geomInst->name = name;
        geomInst->vertexBuffer = vertexBuffer;
        geomInst->triangleBuffer.initialize(optixEnv->cuContext, g_bufferType, triangles);
        geomInst->optixGeomInst = optixEnv->scene.createGeometryInstance();
        geomInst->optixGeomInst.setVertexBuffer(&*vertexBuffer);
        geomInst->optixGeomInst.setTriangleBuffer(&geomInst->triangleBuffer);
        geomInst->optixGeomInst.setNumMaterials(1, nullptr);
        geomInst->optixGeomInst.setMaterial(0, 0, optixEnv->material);
        geomInst->optixGeomInst.setUserData(geomInstIndex);
        geomInst->dataTransfered = false;

        optixEnv->geomInsts[geomInst->serialID] = geomInst;
    }
}




class GeometryInstanceList {
    const OptiXEnv &m_optixEnv;
    bool m_allSelected;
    std::vector<uint32_t> m_selectedItems;

public:
    GeometryInstanceList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv),
        m_allSelected(false) {}

    void show() {
        if (ImGui::BeginTable("##geomInstList", 5,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_ScrollFreezeTopRow,
                              ImVec2(0, 300))) {
            ImGui::TableSetupColumn("CheckAll",
                                    ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_NoResize);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("#Prims", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(ImGui::TableGetColumnName(0));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                if (ImGui::Checkbox("##check", &m_allSelected)) {
                    if (m_allSelected) {
                        for (const auto &kv : m_optixEnv.geomInsts)
                            m_selectedItems.push_back(kv.first);
                    }
                    else {
                        m_selectedItems.clear();
                    }
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                ImGui::TableHeader(ImGui::TableGetColumnName(1));
                ImGui::TableSetColumnIndex(2);
                ImGui::TableHeader(ImGui::TableGetColumnName(2));
                ImGui::TableSetColumnIndex(3);
                ImGui::TableHeader(ImGui::TableGetColumnName(3));
                ImGui::TableSetColumnIndex(4);
                ImGui::TableHeader(ImGui::TableGetColumnName(4));
            }
            for (const auto &kv : m_optixEnv.geomInsts) {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(kv.first);
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                auto itemIndex = std::find(m_selectedItems.cbegin(), m_selectedItems.cend(), kv.first);
                bool selected = itemIndex != m_selectedItems.cend();
                if (ImGui::Checkbox("##check", &selected)) {
                    if (selected)
                        m_selectedItems.push_back(kv.first);
                    else
                        m_selectedItems.erase(itemIndex);
                    m_allSelected = m_selectedItems.size() == m_optixEnv.geomGroups.size();
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                char sid[32];
                sprintf_s(sid, "%u", kv.first);
                ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", kv.second->name.c_str());

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%u", kv.second->triangleBuffer.numElements());

                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%u", kv.second.use_count() - 1);
            }
            ImGui::EndTable();
        }
    }

    uint32_t getNumSelected() const {
        return static_cast<uint32_t>(m_selectedItems.size());
    }
    void loopForSelected(const std::function<bool(uint32_t, const GeometryInstanceRef &)> &func) {
        for (int i = 0; i < m_selectedItems.size(); ++i) {
            bool b = func(i, m_optixEnv.geomInsts.at(m_selectedItems[i]));
            if (!b)
                break;
        }
    }

    const GeometryInstanceRef &getFirstSelectedItem() const {
        return m_optixEnv.geomInsts.at(*m_selectedItems.cbegin());
    }
    void clearSelection() {
        m_allSelected = false;
        m_selectedItems.clear();
    }
};

class GeometryGroupList {
    const OptiXEnv &m_optixEnv;
    bool m_allSelected;
    std::set<uint32_t> m_selectedItems;

public:
    GeometryGroupList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv),
        m_allSelected(false) {}

    void show() {
        if (ImGui::BeginTable("##geomGroupList", 5,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_ScrollFreezeTopRow,
                              ImVec2(0, 300))) {
            ImGui::TableSetupColumn("CheckAll",
                                    ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_NoResize);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("#GeomInsts", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(ImGui::TableGetColumnName(0));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                if (ImGui::Checkbox("##check", &m_allSelected)) {
                    if (m_allSelected) {
                        for (const auto &kv : m_optixEnv.geomGroups)
                            m_selectedItems.insert(kv.first);
                    }
                    else {
                        m_selectedItems.clear();
                    }
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                ImGui::TableHeader(ImGui::TableGetColumnName(1));
                ImGui::TableSetColumnIndex(2);
                ImGui::TableHeader(ImGui::TableGetColumnName(2));
                ImGui::TableSetColumnIndex(3);
                ImGui::TableHeader(ImGui::TableGetColumnName(3));
                ImGui::TableSetColumnIndex(4);
                ImGui::TableHeader(ImGui::TableGetColumnName(4));
            }
            for (const auto &kv : m_optixEnv.geomGroups) {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(kv.first);
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                bool selected = m_selectedItems.count(kv.first) > 0;
                if (ImGui::Checkbox("##check", &selected)) {
                    if (selected)
                        m_selectedItems.insert(kv.first);
                    else
                        m_selectedItems.erase(kv.first);
                    m_allSelected = m_selectedItems.size() == m_optixEnv.geomInsts.size();
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                char sid[32];
                sprintf_s(sid, "%u", kv.first);
                ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", kv.second->name.c_str());

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%u", static_cast<uint32_t>(kv.second->geomInsts.size()));

                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%u", kv.second.use_count() - 1);
            }
            ImGui::EndTable();
        }
    }

    uint32_t getNumSelected() const {
        return static_cast<uint32_t>(m_selectedItems.size());
    }
    void loopForSelected(const std::function<bool(const GeometryGroupRef &)> &func) {
        for (const auto &sid : m_selectedItems) {
            bool b = func(m_optixEnv.geomGroups.at(sid));
            if (!b)
                break;
        }
    }

    const GeometryGroupRef &getFirstSelectedItem() const {
        return m_optixEnv.geomGroups.at(*m_selectedItems.cbegin());
    }
    void clearSelection() {
        m_allSelected = false;
        m_selectedItems.clear();
    }
};

class InstanceList {
    const OptiXEnv &m_optixEnv;
    bool m_allSelected;
    std::set<uint32_t> m_selectedItems;

public:
    InstanceList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv),
        m_allSelected(false) {}

    void show() {
        if (ImGui::BeginTable("##instList", 5,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_ScrollFreezeTopRow,
                              ImVec2(0, 300))) {

            ImGui::TableSetupColumn("CheckAll",
                                    ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_NoResize);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("GAS", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(ImGui::TableGetColumnName(0));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                if (ImGui::Checkbox("##check", &m_allSelected)) {
                    if (m_allSelected) {
                        for (const auto &kv : m_optixEnv.insts)
                            m_selectedItems.insert(kv.first);
                    }
                    else {
                        m_selectedItems.clear();
                    }
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                ImGui::TableHeader(ImGui::TableGetColumnName(1));
                ImGui::TableSetColumnIndex(2);
                ImGui::TableHeader(ImGui::TableGetColumnName(2));
                ImGui::TableSetColumnIndex(3);
                ImGui::TableHeader(ImGui::TableGetColumnName(3));
                ImGui::TableSetColumnIndex(4);
                ImGui::TableHeader(ImGui::TableGetColumnName(4));
            }
            for (const auto &kv : m_optixEnv.insts) {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(kv.first);
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                bool selected = m_selectedItems.count(kv.first) > 0;
                if (ImGui::Checkbox("##check", &selected)) {
                    if (selected)
                        m_selectedItems.insert(kv.first);
                    else
                        m_selectedItems.erase(kv.first);
                    m_allSelected = m_selectedItems.size() == m_optixEnv.insts.size();
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                char sid[32];
                sprintf_s(sid, "%u", kv.first);
                ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", kv.second->name.c_str());

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%s", kv.second->geomGroup->name.c_str());

                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%u", kv.second.use_count() - 1);
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
        m_allSelected = false;
        m_selectedItems.clear();
    }
};

class GroupList {
    const OptiXEnv &m_optixEnv;
    bool m_allSelected;
    std::set<uint32_t> m_selectedItems;

public:
    GroupList(const OptiXEnv &optixEnv) :
        m_optixEnv(optixEnv),
        m_allSelected(false) {}

    void show() {
        if (ImGui::BeginTable("##groupList", 5,
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_ScrollFreezeTopRow,
                              ImVec2(0, 300))) {
            ImGui::TableSetupColumn("CheckAll",
                                    ImGuiTableColumnFlags_WidthFixed |
                                    ImGuiTableColumnFlags_NoResize);
            ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("#insts", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
            {
                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(ImGui::TableGetColumnName(0));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                if (ImGui::Checkbox("##check", &m_allSelected)) {
                    if (m_allSelected) {
                        for (const auto &kv : m_optixEnv.groups)
                            m_selectedItems.insert(kv.first);
                    }
                    else {
                        m_selectedItems.clear();
                    }
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                ImGui::TableHeader(ImGui::TableGetColumnName(1));
                ImGui::TableSetColumnIndex(2);
                ImGui::TableHeader(ImGui::TableGetColumnName(2));
                ImGui::TableSetColumnIndex(3);
                ImGui::TableHeader(ImGui::TableGetColumnName(3));
                ImGui::TableSetColumnIndex(4);
                ImGui::TableHeader(ImGui::TableGetColumnName(4));
            }
            for (const auto &kv : m_optixEnv.groups) {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(kv.first);
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                bool selected = m_selectedItems.count(kv.first) > 0;
                if (ImGui::Checkbox("##check", &selected)) {
                    if (selected)
                        m_selectedItems.insert(kv.first);
                    else
                        m_selectedItems.erase(kv.first);
                    m_allSelected = m_selectedItems.size() == m_optixEnv.groups.size();
                }
                ImGui::PopStyleVar();
                ImGui::PopID();

                ImGui::TableSetColumnIndex(1);
                char sid[32];
                sprintf_s(sid, "%u", kv.first);
                ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", kv.second->name.c_str());

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%u", static_cast<uint32_t>(kv.second->insts.size()));

                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%u", kv.second.use_count() - 1);
            }
            ImGui::EndTable();
        }
    }

    uint32_t getNumSelected() const {
        return static_cast<uint32_t>(m_selectedItems.size());
    }
    void loopForSelected(const std::function<bool(const GroupRef &)> &func) {
        for (const auto &sid : m_selectedItems) {
            bool b = func(m_optixEnv.groups.at(sid));
            if (!b)
                break;
        }
    }

    const GroupRef &getFirstSelectedItem() const {
        return m_optixEnv.groups.at(*m_selectedItems.cbegin());
    }
    void clearSelection() {
        m_allSelected = false;
        m_selectedItems.clear();
    }
};




static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



int32_t mainFunc(int32_t argc, const char* argv[]) {
    const std::filesystem::path exeDir = getExecutableDirectory();

    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    int32_t renderTargetSizeX = 1280;
    int32_t renderTargetSizeY = 720;

    // JP: ウインドウの初期化。
    //     HiDPIディスプレイに対応する。
    // EN: Initialize a window.
    //     Support Hi-DPI display.
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
    float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow(static_cast<int32_t>(renderTargetSizeX * UIScaling),
                                          static_cast<int32_t>(renderTargetSizeY * UIScaling),
                                          "OptiX Utility - Scene Edit", NULL, NULL);
    glfwSetWindowUserPointer(window, nullptr);
    if (!window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    int32_t curFBWidth;
    int32_t curFBHeight;
    glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_FRAMEBUFFER_SRGB);
    GLTK::errorCheck();

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGuiStyle guiStyle, guiStyleWithGamma;
    ImGui::StyleColorsDark(&guiStyle);
    guiStyleWithGamma = guiStyle;
    const auto degamma = [](const ImVec4 &color) {
        return ImVec4(sRGB_degamma_s(color.x),
                      sRGB_degamma_s(color.y),
                      sRGB_degamma_s(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        guiStyleWithGamma.Colors[i] = degamma(guiStyleWithGamma.Colors[i]);
    }
    ImGui::GetStyle() = guiStyleWithGamma;

    io.Fonts->AddFontDefault();

    std::filesystem::path fontPath = exeDir / "fonts/RictyDiminished-Regular.ttf";
    ImFont* fontForFileDialog = io.Fonts->AddFontFromFileTTF(fontPath.u8string().c_str(), 14.0f, nullptr,
                                                             io.Fonts->GetGlyphRangesJapanese());
    if (fontForFileDialog == nullptr)
        hpprintf("Font Not Found!: %s\n", fontPath.u8string().c_str());

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

        switch (button) {
        case GLFW_MOUSE_BUTTON_MIDDLE: {
            devPrintf("Mouse Middle\n");
            g_buttonRotate.recordStateChange(action == GLFW_PRESS, frameIndex);
            break;
        }
        default:
            break;
        }
                               });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        g_mouseX = x;
        g_mouseY = y;
                             });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        switch (key) {
        case GLFW_KEY_W: {
            g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_S: {
            g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_A: {
            g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_D: {
            g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_R: {
            g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_F: {
            g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_Q: {
            g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_E: {
            g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_T: {
            g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_G: {
            g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        default:
            break;
        }
                       });

    g_cameraPositionalMovingSpeed = 0.01f;
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;
    g_cameraPosition = make_float3(0, 0, 3.2f);
    g_cameraOrientation = qRotateY(M_PI);

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUstream cuStream[2];
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[0], 0));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[1], 0));

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false,
                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                DEBUG_SELECT((OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                              OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                              OPTIX_EXCEPTION_FLAG_DEBUG),
                                             OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(exeDir / "scene_edit/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: これらのグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: These are for ray-triangle hit groups, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroup0 = pipeline.createHitProgramGroup(moduleOptiX, RT_CH_NAME_STR("closesthit0"),
                                                                           emptyModule, nullptr,
                                                                           emptyModule, nullptr);

    pipeline.setMaxTraceDepth(1);
    pipeline.link(DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE),
                  false);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    OptiXEnv optixEnv;
    optixEnv.cuContext = cuContext;
    optixEnv.context = optixContext;

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixEnv.material = optixContext.createMaterial();
    optixEnv.material.setHitGroup(Shared::RayType_Primary, hitProgramGroup0);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    constexpr uint32_t MaxNumGeometryInstances = 8192;
    constexpr uint32_t MaxNumGASs = 512;
    
    optixEnv.scene = optixContext.createScene();
    optixEnv.geometryDataBuffer.initialize(cuContext, g_bufferType, MaxNumGeometryInstances);
    optixEnv.geometryInstSlotFinder.initialize(MaxNumGeometryInstances);
    optixEnv.gasDataBuffer.initialize(cuContext, g_bufferType, MaxNumGASs);
    optixEnv.gasSlotFinder.initialize(MaxNumGASs);
    optixEnv.geomInstSerialID = 0;
    optixEnv.gasSerialID = 0;
    optixEnv.instSerialID = 0;
    optixEnv.iasSerialID = 0;
    optixEnv.asScratchBuffer.initialize(cuContext, g_bufferType, 32 * 1024 * 1024, 1);

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    GLTK::Texture2D outputTexture;
    cudau::Array outputArray;
    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputTexture.initialize(renderTargetSizeX, renderTargetSizeY, GLTK::SizedInternalFormat::RGBA32F);
    GLTK::errorCheck();
    outputArray.initializeFromGLTexture2D(cuContext, outputTexture.getRawHandle(),
                                          cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    outputBufferSurfaceHolder.initialize(&outputArray);

    GLTK::Sampler outputSampler;
    outputSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest,
                             GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    GLTK::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    GLTK::GraphicsShader drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "scene_edit/shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "scene_edit/shaders/drawOptiXResult.frag"));



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = 0;
    plp.geomInstData = optixEnv.geometryDataBuffer.getDevicePointer();
    plp.gasData = optixEnv.gasDataBuffer.getDevicePointer();
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

    pipeline.setScene(optixEnv.scene);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));


    
    FileDialog fileDialog;
    fileDialog.setFont(fontForFileDialog);
    fileDialog.setFlags(FileDialog::Flag_FileSelection);
    //fileDialog.setFlags(FileDialog::Flag_FileSelection |
    //                    FileDialog::Flag_DirectorySelection |
    //                    FileDialog::Flag_MultipleSelection);
    
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    bool sbtLayoutUpdated = true;
    uint32_t sbtIndex = -1;
    cudau::Buffer* curShaderBindingTable;
    OptixTraversableHandle curTravHandle = 0;
    while (true) {
        uint32_t bufferIndex = frameIndex % 2;

        if (glfwWindowShouldClose(window))
            break;
        glfwPollEvents();

        bool resized = false;
        int32_t newFBWidth;
        int32_t newFBHeight;
        glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
        if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
            curFBWidth = newFBWidth;
            curFBHeight = newFBHeight;

            renderTargetSizeX = curFBWidth / UIScaling;
            renderTargetSizeY = curFBHeight / UIScaling;
            requestedSize[0] = renderTargetSizeX;
            requestedSize[1] = renderTargetSizeY;

            outputTexture.finalize();
            outputTexture.initialize(renderTargetSizeX, renderTargetSizeY, GLTK::SizedInternalFormat::RGBA32F);
            outputArray.finalize();
            outputArray.initializeFromGLTexture2D(cuContext, outputTexture.getRawHandle(),
                                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            outputArray.resize(renderTargetSizeX, renderTargetSizeY);

            // EN: update the pipeline parameters.
            plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        bool operatingCamera;
        bool cameraIsActuallyMoving;
        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState& a, const KeyState& b) {
                int32_t dir = 0;
                if (a.getState() == true) {
                    if (b.getState() == true)
                        dir = 0;
                    else
                        dir = 1;
                }
                else {
                    if (b.getState() == true)
                        dir = -1;
                    else
                        dir = 0;
                }
                return dir;
            };

            int32_t trackZ = decideDirection(g_keyForward, g_keyBackward);
            int32_t trackX = decideDirection(g_keyLeftward, g_keyRightward);
            int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
            int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

            g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            g_cameraPositionalMovingSpeed = std::min(std::max(g_cameraPositionalMovingSpeed, 1e-6f), 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
            if (g_buttonRotate.getState() == true) {
                if (g_buttonRotate.getTime() == frameIndex) {
                    lastX = g_mouseX;
                    lastY = g_mouseY;
                }
                else {
                    deltaX = g_mouseX - lastX;
                    deltaY = g_mouseY - lastY;
                }
            }

            float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            float3 axis = make_float3(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = make_float3(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation = g_cameraOrientation * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition += g_tempCameraOrientation.toMatrix3x3() * (g_cameraPositionalMovingSpeed * make_float3(trackX, trackY, trackZ));
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == frameIndex) {
                g_cameraOrientation = g_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                               g_keyLeftward.getState() || g_keyRightward.getState() ||
                               g_keyUpward.getState() || g_keyDownward.getState() ||
                               g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                               g_buttonRotate.getState());
            cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 ||
                                      tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;

            plp.camera.position = g_cameraPosition;
            plp.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        CUstream &curCuStream = cuStream[bufferIndex];
        
        // JP: 前フレームの処理が完了するのを待つ。
        // EN: Wait the previous frame processing to finish.
        CUDADRV_CHECK(cuStreamSynchronize(curCuStream));



        {
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&plp.camera.position));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / M_PI;
            rollPitchYaw[1] *= 180 / M_PI;
            rollPitchYaw[2] *= 180 / M_PI;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw, 3))
                g_cameraOrientation = qFromEulerAngles(rollPitchYaw[0] * M_PI / 180,
                                                       rollPitchYaw[1] * M_PI / 180,
                                                       rollPitchYaw[2] * M_PI / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);

            ImGui::End();
        }

        static int32_t travIndex = -1;
        static std::vector<std::string> traversableNames;
        static std::vector<OptixTraversableHandle> traversables;
        bool traversablesUpdated = false;
        {
            ImGui::Begin("Scene", nullptr,
                         ImGuiWindowFlags_None);

            if (ImGui::Button("Open"))
                fileDialog.show();
            if (fileDialog.drawAndGetResult() == FileDialog::Result::Result_OK) {
                static std::vector<std::filesystem::directory_entry> entries;
                fileDialog.calcEntries(&entries);
                
                loadFile(entries[0], &optixEnv);
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
                const auto ImGui_PushDisabledStyle = []() {
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.2f);
                };
                const auto ImGui_PopDisabledStyle = []() {
                    ImGui::PopStyleVar();
                };

                if (ImGui::BeginTabItem("Geom Inst")) {
                    static GeometryInstanceList geomInstList(optixEnv);
                    geomInstList.show();

                    bool geomInstsSelected = geomInstList.getNumSelected() > 0;
                    bool allUnused = geomInstList.getNumSelected() > 0;
                    geomInstList.loopForSelected(
                        [&allUnused](uint32_t idx, const GeometryInstanceRef &geomInst) {
                            allUnused &= geomInst.use_count() == 1;
                            return allUnused;
                        });
                    bool selectedGeomInstsRemovable = allUnused && geomInstList.getNumSelected() > 0;

                    if (!geomInstsSelected)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create a GAS")) {
                        if (geomInstsSelected) {
                            uint32_t serialID = optixEnv.gasSerialID++;
                            GeometryGroupRef geomGroup = make_shared_with_deleter<GeometryGroup>(GeometryGroup::finalize);
                            uint32_t gasIndex = optixEnv.gasSlotFinder.getFirstAvailableSlot();
                            optixEnv.gasSlotFinder.setInUse(gasIndex);
                            char name[256];
                            sprintf_s(name, "GAS-%u", serialID);
                            geomGroup->optixEnv = &optixEnv;
                            geomGroup->gasIndex = gasIndex;
                            geomGroup->serialID = serialID;
                            geomGroup->name = name;
                            geomGroup->optixGAS = optixEnv.scene.createGeometryAccelerationStructure();
                            geomGroup->optixGAS.setConfiguration(false, false, false, false);
                            geomGroup->optixGAS.setNumMaterialSets(1);
                            geomGroup->optixGAS.setNumRayTypes(0, Shared::NumRayTypes);
                            geomGroup->preTransformBuffer.initialize(optixEnv.cuContext, g_bufferType, geomInstList.getNumSelected());
                            geomGroup->dataTransfered = false;

                            geomInstList.loopForSelected(
                                [&optixEnv, &geomGroup, &curCuStream](uint32_t idx, const GeometryInstanceRef &geomInst) {
                                    geomGroup->geomInsts.push_back(geomInst);
                                    geomGroup->preTransforms.emplace_back();
                                    geomGroup->optixGAS.addChild(geomInst->optixGeomInst, geomGroup->preTransformBuffer.getCUdeviceptrAt(idx));
                                    if (!geomInst->dataTransfered) {
                                        Shared::GeometryData geomData;
                                        geomData.vertexBuffer = geomInst->vertexBuffer->getDevicePointer();
                                        geomData.triangleBuffer = geomInst->triangleBuffer.getDevicePointer();
                                        CUDADRV_CHECK(cuMemcpyHtoDAsync(optixEnv.geometryDataBuffer.getCUdeviceptrAt(geomInst->geomInstIndex),
                                                                        &geomData, sizeof(geomData), curCuStream));
                                        geomInst->dataTransfered = true;
                                    }
                                    return true;
                                });
                            CUDADRV_CHECK(cuMemcpyHtoDAsync(geomGroup->preTransformBuffer.getCUdeviceptr(),
                                                            geomGroup->preTransforms.data(),
                                                            geomGroup->preTransformBuffer.sizeInBytes(),
                                                            curCuStream));

                            sbtLayoutUpdated = true;
                            traversablesUpdated = true;

                            optixEnv.geomGroups[serialID] = geomGroup;
                        }
                    }
                    if (!geomInstsSelected)
                        ImGui_PopDisabledStyle();

                    ImGui::SameLine();
                    if (!selectedGeomInstsRemovable)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (selectedGeomInstsRemovable) {
                            geomInstList.loopForSelected(
                                [&optixEnv](uint32_t idx, const GeometryInstanceRef &geomInst) {
                                    optixEnv.geomInsts.erase(geomInst->serialID);
                                    return true;
                                });
                            geomInstList.clearSelection();
                        }
                    }
                    if (!selectedGeomInstsRemovable)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Geom Group")) {
                    static GeometryGroupList geomGroupList(optixEnv);
                    geomGroupList.show();

                    bool singleGeomGroupSelected = geomGroupList.getNumSelected() == 1;
                    bool allUnused = geomGroupList.getNumSelected() > 0;
                    geomGroupList.loopForSelected(
                        [&allUnused](const GeometryGroupRef &geomGroup) {
                            allUnused &= geomGroup.use_count() == 1;
                            return allUnused;
                        });
                    bool selectedGeomGroupsRemovable = allUnused && geomGroupList.getNumSelected() > 0;

                    if (!singleGeomGroupSelected)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create an Instance")) {
                        if (singleGeomGroupSelected) {
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

                            const GeometryGroupRef &geomGroup = geomGroupList.getFirstSelectedItem();
                            inst->geomGroup = geomGroup;
                            geomGroup->parentInsts.insert(inst);
                            inst->optixInst = optixEnv.scene.createInstance();
                            inst->optixInst.setGAS(geomGroup->optixGAS);
                            float tr[] = {
                                srMat.m00, srMat.m01, srMat.m02, inst->position.x,
                                srMat.m10, srMat.m11, srMat.m12, inst->position.y,
                                srMat.m20, srMat.m21, srMat.m22, inst->position.z,
                            };
                            inst->optixInst.setTransform(tr);

                            if (!geomGroup->dataTransfered) {
                                Shared::GASData gasData;
                                gasData.preTransforms = geomGroup->preTransformBuffer.getDevicePointer();
                                CUDADRV_CHECK(cuMemcpyHtoDAsync(optixEnv.gasDataBuffer.getCUdeviceptrAt(geomGroup->gasIndex),
                                                                &gasData,
                                                                sizeof(gasData),
                                                                curCuStream));
                                geomGroup->dataTransfered = true;
                            }

                            optixEnv.insts[serialID] = inst;
                        }
                    }
                    if (!singleGeomGroupSelected)
                        ImGui_PopDisabledStyle();

                    ImGui::SameLine();
                    if (!selectedGeomGroupsRemovable)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove##GeomGroup")) {
                        if (selectedGeomGroupsRemovable) {
                            geomGroupList.loopForSelected(
                                [&optixEnv](const GeometryGroupRef &geomGroup) {
                                    optixEnv.geomGroups.erase(geomGroup->serialID);
                                    return true;
                                });
                            geomGroupList.clearSelection();
                            sbtLayoutUpdated = true;
                            traversablesUpdated = true;
                        }
                    }
                    if (!selectedGeomGroupsRemovable)
                        ImGui_PopDisabledStyle();



                    ImGui::Separator();

                    if (!singleGeomGroupSelected)
                        ImGui_PushDisabledStyle();

                    static GeometryGroupRef dummyGeomGroup = make_shared_with_deleter<GeometryGroup>(GeometryGroup::finalize);
                    const GeometryGroupRef &selectedGeomGroup = singleGeomGroupSelected ?
                        geomGroupList.getFirstSelectedItem() : dummyGeomGroup;

                    ImGui::Text("%s", singleGeomGroupSelected ? selectedGeomGroup->name.c_str() : "Not selected");

                    static bool allGeomInstSelected = false;
                    static std::set<uint32_t> selectedGeomInsts;
                    if (ImGui::BeginTable("##geomInstList", 4,
                                          ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_ScrollFreezeTopRow,
                                          ImVec2(0, 200))) {
                        ImGui::TableSetupColumn("CheckAll",
                                                ImGuiTableColumnFlags_WidthFixed |
                                                ImGuiTableColumnFlags_NoResize);
                        ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("#Prims", ImGuiTableColumnFlags_WidthFixed);
                        {
                            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(ImGui::TableGetColumnName(0));
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                            if (ImGui::Checkbox("##check", &allGeomInstSelected)) {
                                if (allGeomInstSelected) {
                                    for (int i = 0; i < selectedGeomGroup->geomInsts.size(); ++i)
                                        selectedGeomInsts.insert(i);
                                }
                                else {
                                    selectedGeomInsts.clear();
                                }
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            ImGui::TableHeader(ImGui::TableGetColumnName(1));
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TableHeader(ImGui::TableGetColumnName(2));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TableHeader(ImGui::TableGetColumnName(3));
                        }
                        for (int i = 0; i < selectedGeomGroup->geomInsts.size(); ++i) {
                            const GeometryInstanceRef &geomInst = selectedGeomGroup->geomInsts[i];

                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(i);
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                            bool selected = selectedGeomInsts.count(i) > 0;
                            if (ImGui::Checkbox("##check", &selected)) {
                                if (selected)
                                    selectedGeomInsts.insert(i);
                                else
                                    selectedGeomInsts.erase(i);
                                allGeomInstSelected = selectedGeomInsts.size() == optixEnv.geomInsts.size();
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            char sid[32];
                            sprintf_s(sid, "%u", geomInst->serialID);
                            ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%s", geomInst->name.c_str());

                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%u", geomInst->triangleBuffer.numElements());
                        }
                        ImGui::EndTable();
                    }

                    bool geomInstsSelected = selectedGeomInsts.size() > 0;

                    if (!singleGeomGroupSelected)
                        ImGui_PopDisabledStyle();

                    if (!geomInstsSelected)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove##GeomInst")) {
                        if (geomInstsSelected) {
                            for (auto it = selectedGeomInsts.crbegin(); it != selectedGeomInsts.crend(); ++it) {
                                const GeometryInstanceRef &child = selectedGeomGroup->geomInsts[*it];
                                selectedGeomGroup->geomInsts.erase(selectedGeomGroup->geomInsts.cbegin() + *it);
                                selectedGeomGroup->optixGAS.removeChild(child->optixGeomInst);
                            }

                            sbtLayoutUpdated = true;
                            traversablesUpdated = true;

                            selectedGeomInsts.clear();
                            allGeomInstSelected = false;
                        }
                    }
                    if (!geomInstsSelected)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Inst")) {
                    static InstanceList instList(optixEnv);
                    instList.show();

                    bool instsSelected = instList.getNumSelected() > 0;
                    bool singleInstSelected = instList.getNumSelected() == 1;
                    bool allUnused = instList.getNumSelected() > 0;
                    instList.loopForSelected(
                        [&allUnused](const InstanceRef &inst) {
                            allUnused &= inst.use_count() == 1;
                            return allUnused;
                        });
                    bool selectedInstsRemovable = allUnused && instList.getNumSelected() > 0;

                    if (!instsSelected)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create an IAS")) {
                        if (instsSelected) {
                            uint32_t serialID = optixEnv.iasSerialID++;
                            GroupRef group = make_shared_with_deleter<Group>(Group::finalize);
                            group->optixEnv = &optixEnv;
                            char name[256];
                            sprintf_s(name, "IAS-%u", serialID);
                            group->serialID = serialID;
                            group->name = name;
                            group->optixIAS = optixEnv.scene.createInstanceAccelerationStructure();
                            group->optixIAS.setConfiguration(false, false, false);

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
                    }
                    if (!instsSelected)
                        ImGui_PopDisabledStyle();

                    ImGui::SameLine();
                    if (!selectedInstsRemovable)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (selectedInstsRemovable) {
                            instList.loopForSelected(
                                [&optixEnv](const InstanceRef &inst) {
                                    InstanceWRef instWRef = std::weak_ptr(inst);
                                    inst->geomGroup->parentInsts.erase(instWRef);
                                    optixEnv.insts.erase(inst->serialID);
                                    return true;
                                });
                            instList.clearSelection();
                        }
                    }
                    if (!selectedInstsRemovable)
                        ImGui_PopDisabledStyle();



                    ImGui::Separator();
                    if (!singleInstSelected)
                        ImGui_PushDisabledStyle();

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
                        instOrientation[0] *= 180 / M_PI;
                        instOrientation[1] *= 180 / M_PI;
                        instOrientation[2] *= 180 / M_PI;
                        std::copy_n(reinterpret_cast<float*>(&selectedInst->position), 3, instPosition);
                    }
                    else {
                        instScale[0] = instScale[1] = instScale[2] = 0.0f;
                        instOrientation[0] = instOrientation[1] = instOrientation[2] = 0.0f;
                        instPosition[0] = instPosition[1] = instPosition[2] = 0.0f;
                    }
                    srtUpdated |= ImGui::InputFloat3("Scale", instScale, 5);
                    srtUpdated |= ImGui::InputFloat3("Roll/Pitch/Yaw", instOrientation, 5);
                    srtUpdated |= ImGui::InputFloat3("Position", instPosition, 5);
                    if (singleInstSelected && srtUpdated) {
                        selectedInst->scale = float3(instScale[0], instScale[1], instScale[2]);
                        selectedInst->rollPitchYaw[0] = instOrientation[0] * M_PI / 180;
                        selectedInst->rollPitchYaw[1] = instOrientation[1] * M_PI / 180;
                        selectedInst->rollPitchYaw[2] = instOrientation[2] * M_PI / 180;
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

                        for (const auto &parent : selectedInst->parentGroups) {
                            GroupRef group = parent.lock();
                            group->optixIAS.markDirty();
                        }
                    }

                    if (!singleInstSelected)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Group")) {
                    static GroupList groupList(optixEnv);
                    groupList.show();

                    bool groupsSelected = groupList.getNumSelected() > 0;
                    if (!groupsSelected)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (groupsSelected) {
                            groupList.loopForSelected(
                                [&optixEnv](const GroupRef &group) {
                                    GroupWRef groupWRef = std::weak_ptr(group);
                                    for (const auto &child : group->insts)
                                        child->parentGroups.erase(groupWRef);
                                    optixEnv.groups.erase(group->serialID);
                                    return true;
                                });
                            groupList.clearSelection();
                            traversablesUpdated = true;
                        }
                    }
                    if (!groupsSelected)
                        ImGui_PopDisabledStyle();

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

            // TODO: GAS用バッファーのアップデートもレンダリングで使いながらのアップデートは危険？
            OptixAccelBufferSizes bufferSizes;
            geomGroup->optixGAS.prepareForBuild(&bufferSizes);
            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1, curCuStream);
            if (geomGroup->optixGasMem.isInitialized())
                geomGroup->optixGasMem.resize(bufferSizes.outputSizeInBytes, 1, curCuStream);
            else
                geomGroup->optixGasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
            geomGroup->optixGAS.rebuild(curCuStream, geomGroup->optixGasMem, optixEnv.asScratchBuffer);
        }

        if (sbtLayoutUpdated) {
            sbtIndex = (sbtIndex + 1) % 2;
            curShaderBindingTable = &optixEnv.shaderBindingTable[sbtIndex];

            size_t sbtSize;
            optixEnv.scene.generateShaderBindingTableLayout(&sbtSize);
            if (curShaderBindingTable->isInitialized())
                curShaderBindingTable->resize(sbtSize, 1, curCuStream);
            else
                curShaderBindingTable->initialize(cuContext, g_bufferType, sbtSize, 1);
            pipeline.setHitGroupShaderBindingTable(curShaderBindingTable);
            sbtLayoutUpdated = false;
        }

        for (const auto &kv : optixEnv.groups) {
            const GroupRef &group = kv.second;
            if (group->optixIAS.isReady())
                continue;

            // TODO: 非同期実行中のスクラッチバッファー更新大丈夫？
            //       連続フレームで来ると危ない？同じフレーム内でもGAS, IASと連続してサイズ変更が危ない？
            OptixAccelBufferSizes bufferSizes;
            uint32_t numInstances;
            group->optixIAS.prepareForBuild(&bufferSizes, &numInstances);
            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1, curCuStream);
            if (group->optixIasMem.isInitialized()) {
                group->optixIasMem.resize(bufferSizes.outputSizeInBytes, 1, curCuStream);
                group->optixInstanceBuffer.resize(numInstances);
            }
            else {
                group->optixIasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
                group->optixInstanceBuffer.initialize(optixEnv.cuContext, g_bufferType, numInstances);
            }
            group->optixIAS.rebuild(curCuStream, group->optixInstanceBuffer, group->optixIasMem, optixEnv.asScratchBuffer);
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
        outputBufferSurfaceHolder.beginCUDAAccess(curCuStream);

        plp.travHandle = curTravHandle;
        plp.resultBuffer = outputBufferSurfaceHolder.getNext();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curCuStream));
        pipeline.launch(curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);

        outputBufferSurfaceHolder.endCUDAAccess(curCuStream);



        // ----------------------------------------------------------------
        // JP: 

        glEnable(GL_FRAMEBUFFER_SRGB);
        GLTK::errorCheck();

        glViewport(0, 0, curFBWidth, curFBHeight);

        drawOptiXResultShader.useProgram();

        glUniform2ui(0, curFBWidth, curFBHeight);

        glActiveTexture(GL_TEXTURE0);
        outputTexture.bind();
        outputSampler.bindToTextureUnit(0);

        vertexArrayForFullScreen.bind();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        vertexArrayForFullScreen.unbind();

        outputTexture.unbind();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // END: 
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);

        ++frameIndex;
    }

    CUDADRV_CHECK(cuStreamSynchronize(cuStream[frameIndex % 2]));

    CUDADRV_CHECK(cuMemFree(plpOnDevice));

    optixEnv.groups.clear();
    optixEnv.insts.clear();
    optixEnv.geomGroups.clear();
    optixEnv.geomInsts.clear();

    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();

    optixEnv.asScratchBuffer.finalize();
    optixEnv.shaderBindingTable[1].finalize();
    optixEnv.shaderBindingTable[0].finalize();
    optixEnv.gasSlotFinder.finalize();
    optixEnv.gasDataBuffer.finalize();
    optixEnv.geometryInstSlotFinder.finalize();
    optixEnv.geometryDataBuffer.finalize();
    optixEnv.scene.destroy();

    optixEnv.material.destroy();

    hitProgramGroup0.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(cuStream[1]));
    CUDADRV_CHECK(cuStreamDestroy(cuStream[0]));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    
    glfwTerminate();

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
    }

    return 0;
}
