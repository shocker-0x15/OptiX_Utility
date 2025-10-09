#include "imgui_file_dialog.h"

FileDialog::Result FileDialog::drawAndGetResult() {
    namespace fs = std::filesystem;

    if (m_font)
        ImGui::PushFont(m_font);

    ImGuiIO &io = ImGui::GetIO();
    ImGuiStyle &style = ImGui::GetStyle();

    const auto getSeparatorHeightWithSpacing = [&style]() {
        return std::fmax(1.0f, style.ItemSpacing.y);
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

        float fileListHeight = std::fmax(ImGui::GetWindowSize().y - windowMinHeight, 0.0f) + fileListMinHeight;
        ImGui::BeginChild("File List", ImVec2(ImGui::GetContentRegionAvail().x, fileListHeight));

        bool multiplySelected = (m_numSelectedDirs + m_numSelectedFiles) > 1;
        bool selectionChanged = false;
        for (int i = 0; i < m_entryInfos.size(); ++i) {
            EntryInfo &entryInfo = m_entryInfos[i];

            std::string name = entryInfo.is_directory() ? "[D] " : "[F] ";
            name += reinterpret_cast<const char*>(entryInfo.path().filename().u8string().c_str());
            if (ImGui::Selectable(name.c_str(), entryInfo.selected, ImGuiSelectableFlags_DontClosePopups)) {
                if (((m_flags & Flag_DirectorySelection) && entryInfo.is_directory()) ||
                    ((m_flags & Flag_FileSelection) && !entryInfo.is_directory())) {
                    // JP: 単一選択した場合は他の選択済み項目を非選択状態に変更する。
                    if (!ImGui::GetIO().KeyCtrl || ((m_flags & Flag_MultipleSelection) == 0)) {
                        for (EntryInfo &e : m_entryInfos) {
                            // JP: 複数選択状態なら選択状態になるようにする。
                            if (&e != &entryInfo || multiplySelected)
                                e.selected = false;
                        }
                    }
                    entryInfo.selected ^= true;
                    selectionChanged = true;
                }
            }
            // JP: ダブルクリック時の処理。
            if (ImGui::IsMouseDoubleClicked(0) && !ImGui::GetIO().KeyCtrl && ImGui::IsItemHovered()) {
                if (entryInfo.is_directory()) {
                    if (entryInfo.path().is_relative())
                        newDir = m_curDir / entryInfo.path();
                    else
                        newDir = entryInfo.path();
                }
                else {
                    // JP: 他の選択済み項目を非選択状態に変更する。
                    for (EntryInfo &e : m_entryInfos) {
                        // JP: 複数選択状態なら自身だけ選択状態になるようにする。
                        if (&e != &entryInfo || multiplySelected)
                            e.selected = false;
                    }
                    if ((m_flags & Flag_FileSelection) && !entryInfo.is_directory())
                        fileDoubleClicked = true;
                    entryInfo.selected = true;
                    selectionChanged = true;
                }
            }
        }

        if (selectionChanged) {
            m_numSelectedFiles = 0;
            m_numSelectedDirs = 0;
            for (int i = 0; i < m_entryInfos.size(); ++i) {
                const EntryInfo &entryInfo = m_entryInfos[i];
                if (entryInfo.selected) {
                    if (entryInfo.is_directory())
                        ++m_numSelectedDirs;
                    else
                        ++m_numSelectedFiles;
                }
            }
            genFilesText();
        }

        ImGui::EndChild();

        // END: 
        // ----------------------------------------------------------------

        ImGui::Separator();

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(reducedItemSpacing, reducedItemSpacing));
        ImGui::Text("%u files, %u dirs selected", m_numSelectedFiles, m_numSelectedDirs);
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

    entries->clear();

    std::string filesText = m_curFilesText;
    std::vector<std::string> fileFilters;
    while (!filesText.empty()) {
        size_t p = filesText.find(',');
        std::string file;
        if (p == std::string::npos) {
            file = filesText;
            filesText = "";
        }
        else {
            file = filesText.substr(0, p);
            filesText = filesText.substr(p + 1);
        }
        if (!file.empty())
            fileFilters.push_back(file);
    }

    for (const std::string &filter : fileFilters) {
        for (const EntryInfo &entryInfo : m_entryInfos) {
            if ((((m_flags & Flag_DirectorySelection) == 0) && entryInfo.is_directory()) ||
                (((m_flags & Flag_FileSelection) == 0) && !entryInfo.is_directory()))
                continue;

            if (reinterpret_cast<const char*>(entryInfo.path().filename().u8string().c_str()) == filter) {
                if (entryInfo.path().is_relative())
                    entries->emplace_back(fs::canonical(m_curDir / entryInfo.path()));
                else
                    entries->push_back(entryInfo.entry);
                break;
            }
        }
    }
}
