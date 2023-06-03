#pragma once

#include "imgui.h"
#include <algorithm>

namespace ImGui {
    template <typename EnumType>
    bool RadioButtonE(const char* label, EnumType* v, EnumType v_button) {
        return RadioButton(label, reinterpret_cast<int*>(v), static_cast<int>(v_button));
    }

    inline bool InputLog2Int(const char* label, int* v, int max_v, int num_digits = 3) {
        float buttonSize = GetFrameHeight();
        float itemInnerSpacingX = GetStyle().ItemInnerSpacing.x;

        BeginGroup();
        PushID(label);

        ImGui::AlignTextToFramePadding();
        SetNextItemWidth(std::max(1.0f, CalcItemWidth() - (buttonSize + itemInnerSpacingX) * 2));
        Text("%s: %*u", label, num_digits, 1 << *v);
        bool changed = false;
        SameLine(0, itemInnerSpacingX);
        if (Button("-", ImVec2(buttonSize, buttonSize))) {
            *v = std::max(*v - 1, 0);
            changed = true;
        }
        SameLine(0, itemInnerSpacingX);
        if (Button("+", ImVec2(buttonSize, buttonSize))) {
            *v = std::min(*v + 1, max_v);
            changed = true;
        }

        PopID();
        EndGroup();

        return changed;
    }

    inline void PushDisabledStyle() {
        PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.2f);
    }
    inline void PopDisabledStyle() {
        PopStyleVar();
    }

    inline bool Button(const char* label, bool active, const ImVec2 &size = ImVec2(0, 0)) {
        if (!active)
            PushDisabledStyle();
        bool ret = Button(label, size) && active;
        if (!active)
            PopDisabledStyle();
        return ret;
    }
}
