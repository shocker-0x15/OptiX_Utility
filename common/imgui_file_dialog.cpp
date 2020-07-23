#include "imgui_file_dialog.h"

FileDialog::Result FileDialog::drawAndGetResult() {
    namespace fs = std::filesystem;

    if (m_font)
        ImGui::PushFont(m_font);

    ImGuiIO &io = ImGui::GetIO();
    ImGuiStyle &style = ImGui::GetStyle();

    const auto getSeparatorHeightWithSpacing = [&style]() {
        return std::fmax(1, style.ItemSpacing.y);
    };

    Result res = Result_Undecided;
    const float fileListMinHeight = 100;
    const float reducedItemSpacing = 1;
    const float windowMinHeight =
        ImGui::GetFrameHeight() + // Title bar
        ImGui::GetFrameHeightWithSpacing() + // dir path
        ImGui::GetFrameHeightWithSpacing() + // dir path buttons
        getSeparatorHeightWithSpacing() + // separator
        fileListMinHeight + style.ItemSpacing.y +
        getSeparatorHeightWithSpacing() + // separator
        ImGui::GetTextLineHeight() + reducedItemSpacing + // select count
        ImGui::GetFrameHeightWithSpacing() + // cur files
        ImGui::GetFrameHeight() + // OK/Cancel buttons
        2 * style.WindowPadding.y;
    //ImGui::SetNextWindowSize(ImVec2(400, -1));
    ImGui::SetNextWindowSizeConstraints(ImVec2(400, windowMinHeight), ImVec2(1e+6, 1e+6));
    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (ImGui::BeginPopupModal(m_title, nullptr)) {
        if (m_curDir.empty()) {
            changeDirectory(fs::current_path());
        }

        fs::path newDir;
        bool fileDoubleClicked = false;

        // JP: カレントディレクトリの文字列を表示。
        // EN: 
        ImGui::PushID("Directory");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("", m_curDirText, sizeof(m_curDirText), ImGuiInputTextFlags_EnterReturnsTrue)) {
            fs::path dir = m_curDirText;
            if (fs::exists(dir)) {
                if (!fs::is_directory(dir))
                    dir.remove_filename();
                newDir = dir;
            }
            else {
                newDir = m_curDir;
            }
        }
        ImGui::PopID();

        // JP: カレントディレクトリのパスを分解してボタンとして表示する。
        // EN: 
        //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::BeginChild("Path Buttons", ImVec2(0, ImGui::GetFrameHeight()));
        for (auto it = m_curDirBlocks.begin(); it != m_curDirBlocks.end(); ++it) {
            if (it != m_curDirBlocks.begin())
                ImGui::SameLine();
            if (ImGui::Button(it->c_str())) {
                for (auto nit = m_curDirBlocks.begin(); nit != m_curDirBlocks.end(); ++nit) {
                    newDir /= *nit;
                    if (nit->find(':') == nit->length() - 1)
                        newDir /= "\\";
                    if (nit == it)
                        break;
                }
            }
        }
        ImGui::EndChild();
        //ImGui::PopStyleVar();

        ImGui::Separator();

        // ----------------------------------------------------------------
        // JP: ファイルリストの表示と選択処理。

        float fileListHeight = std::fmax(ImGui::GetWindowSize().y - windowMinHeight, 0) + fileListMinHeight;
        ImGui::BeginChild("File List", ImVec2(ImGui::GetWindowContentRegionWidth(), fileListHeight));

        bool selectionChanged = false;
        for (int i = 0; i < m_files.size(); ++i) {
            uint8_t &selected = m_fileSelectedStates[i];
            const fs::directory_entry &entry = m_files[i];

            std::string name = entry.is_directory() ? "[D] " : "[F] ";
            name += entry.path().filename().u8string().c_str();
            if (ImGui::Selectable(name.c_str(), selected, ImGuiSelectableFlags_DontClosePopups)) {
                if (!ImGui::GetIO().KeyCtrl || ((m_flags & Flag_MultipleSelection) == 0)) {
                    for (uint8_t &s : m_fileSelectedStates) {
                        if (&s != &selected || m_multiplySelected)
                            s = false;
                    }
                }
                if (((m_flags & Flag_DirectorySelection) && entry.is_directory()) ||
                    ((m_flags & Flag_FileSelection) && !entry.is_directory())) {
                    selected ^= true;
                    selectionChanged = true;
                }
            }
            if (ImGui::IsMouseDoubleClicked(0) && ImGui::IsItemHovered()) {
                if (entry.is_directory()) {
                    if (!ImGui::GetIO().KeyCtrl) {
                        if (entry.path().is_relative())
                            newDir = m_curDir / entry.path();
                        else
                            newDir = entry.path();
                    }
                }
                else {
                    if (!ImGui::GetIO().KeyCtrl) {
                        for (uint8_t &s : m_fileSelectedStates) {
                            if (&s != &selected || m_multiplySelected)
                                s = false;
                        }
                        fileDoubleClicked = true;
                    }
                    selected = true;
                    selectionChanged = true;
                }
            }
        }

        if (selectionChanged)
            genFilesText();

        ImGui::EndChild();

        // END: 
        // ----------------------------------------------------------------

        ImGui::Separator();

        uint32_t numSelectedFiles = 0;
        uint32_t numSelectedDirs = 0;
        for (int i = 0; i < m_files.size(); ++i) {
            const uint8_t &selected = m_fileSelectedStates[i];
            const fs::directory_entry &entry = m_files[i];
            if (selected) {
                if (entry.is_directory())
                    ++numSelectedDirs;
                else
                    ++numSelectedFiles;
            }
        }
        m_multiplySelected = (numSelectedDirs + numSelectedFiles) > 1;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(reducedItemSpacing, reducedItemSpacing));
        ImGui::Text("%u files, %u dirs selected", numSelectedFiles, numSelectedDirs);
        ImGui::PopStyleVar();

        ImGui::PushID("Current Files");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("", m_curFilesText, sizeof(m_curFilesText))) {

        }
        ImGui::PopID();

        if (!newDir.empty())
            changeDirectory(newDir);

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            res = Result_Cancel;
        }
        ImGui::SameLine();
        if (ImGui::Button("OK") || fileDoubleClicked) {
            ImGui::CloseCurrentPopup();
            res = Result_OK;
        }

        ImGui::EndPopup();
    }

    if (m_font)
        ImGui::PopFont();

    return res;
}

void FileDialog::calcEntries(std::vector<std::filesystem::directory_entry>* entries) {
    namespace fs = std::filesystem;

    if (ImGui::IsPopupOpen(m_title))
        entries->clear();

    std::string filesText = m_curFilesText;
    std::vector<std::string> fileFilters;
    while (!filesText.empty()) {
        size_t p = filesText.find(',');
        std::string newFilter;
        if (p == std::string::npos) {
            newFilter = filesText;
            filesText = "";
        }
        else {
            newFilter = filesText.substr(0, p);
            filesText = filesText.substr(p + 1);
        }
        if (!newFilter.empty())
            fileFilters.push_back(newFilter);
    }

    for (const std::string &filter : fileFilters) {
        for (const fs::directory_entry &entry : m_files) {
            if (((m_flags & Flag_DirectorySelection == 0) && entry.is_directory()) ||
                ((m_flags & Flag_FileSelection == 0) && !entry.is_directory()))
                continue;

            if (entry.path().filename().u8string() == filter) {
                if (entry.path().is_relative())
                    entries->emplace_back(fs::canonical(m_curDir / entry.path()));
                else
                    entries->push_back(entry);
                break;
            }
        }
    }
}
