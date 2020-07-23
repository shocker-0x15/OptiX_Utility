#pragma once

#include "imgui.h"
#include <filesystem>
#include <vector>

class FileDialog {
public:
    enum Flag {
        Flag_FileSelection = 1 << 0,
        Flag_DirectorySelection = 1 << 1,
        Flag_MultipleSelection = 1 << 2,
    };

    enum Result {
        Result_Undecided = 0,
        Result_OK,
        Result_Cancel
    };

private:
    std::filesystem::path m_curDir;
    std::vector<std::string> m_curDirBlocks;
    std::vector<std::filesystem::directory_entry> m_files;
    std::vector<uint8_t> m_fileSelectedStates;
    char m_curDirText[1024];
    char m_curFilesText[1024];
    bool m_multiplySelected;
    const char* m_title;
    ImFont* m_font;
    Flag m_flags;

    template <size_t Size>
    static void strToChars(const std::string &src, char(&dst)[Size]) {
        strncpy_s(dst, src.c_str(), Size - 1);
        dst[Size - 1] = '\0';
    }
    template <size_t Size>
    static void pathToChars(const std::filesystem::path &path, char(&dst)[Size]) {
        strncpy_s(dst, path.u8string().c_str(), Size - 1);
        dst[Size - 1] = '\0';
    }

    void genFilesText() {
        namespace fs = std::filesystem;

        std::string text;
        bool firstFile = true;
        for (int i = 0; i < m_files.size(); ++i) {
            const uint8_t &selected = m_fileSelectedStates[i];
            const fs::directory_entry &entry = m_files[i];
            if (!selected)
                continue;

            if (!firstFile)
                text += ",";
            text += entry.path().filename().u8string();
            firstFile = false;
        }

        strToChars(text, m_curFilesText);
    }

    void changeDirectory(const std::filesystem::path &dirPath) {
        namespace fs = std::filesystem;

        m_curDir = fs::canonical(dirPath);

        m_curDirBlocks.clear();
        for (auto it = m_curDir.begin(); it != m_curDir.end(); ++it) {
            std::string str = it->u8string();
            if (strcmp(str.c_str(), "\\") == 0) {
                continue;
            }
            m_curDirBlocks.push_back(str);
        }

        m_files.clear();
        m_fileSelectedStates.clear();
        fs::directory_entry up{ ".." };
        m_files.push_back(up);
        m_fileSelectedStates.push_back(false);
        m_multiplySelected = false;
        for (const auto &entry : fs::directory_iterator(m_curDir)) {
            m_files.push_back(entry);
            m_fileSelectedStates.push_back(false);
        }

        pathToChars(m_curDir, m_curDirText);
        genFilesText();
    }

public:
    FileDialog() : m_curDir(""), m_font(nullptr) {
        m_title = "Default Title";
    }

    void setFont(ImFont* font) {
        m_font = font;
    }

    void setFlags(uint32_t flags) {
        m_flags = static_cast<Flag>(flags);
    }

    void show() {
        ImGui::OpenPopup(m_title);
    }

    Result drawAndGetResult();

    void calcEntries(std::vector<std::filesystem::directory_entry>* entries);
};
