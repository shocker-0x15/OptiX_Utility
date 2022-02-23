#pragma once

#include "imgui.h"
#include <filesystem>
#include <vector>

class FileDialog {
public:
    enum Flag {
        Flag_FileSelection      = 1 << 0,
        Flag_DirectorySelection = 1 << 1,
        Flag_MultipleSelection  = 1 << 2,
    };

    enum Result {
        Result_Undecided = 0,
        Result_OK,
        Result_Cancel
    };

private:
    struct EntryInfo {
        std::filesystem::directory_entry entry;
        union {
            uint32_t flags;
            struct {
                unsigned int selected : 1;
            };
        };
        EntryInfo() : flags(0) {}

        bool is_directory() const {
            return entry.is_directory();
        }
        const std::filesystem::path &path() const noexcept {
            return entry.path();
        }
    };

    std::filesystem::path m_curDir;
    std::vector<std::string> m_curDirBlocks;
    std::vector<EntryInfo> m_entryInfos;
    uint32_t m_numSelectedFiles;
    uint32_t m_numSelectedDirs;
    char m_curDirText[1024];
    char m_curFilesText[1024];
    const char* m_title;
    ImFont* m_font;
    Flag m_flags;

    template <size_t Size>
    static void strToChars(const std::string &src, char (&dst)[Size]) {
        strncpy_s(dst, src.c_str(), Size - 1);
        dst[Size - 1] = '\0';
    }
    template <size_t Size>
    static void pathToChars(const std::filesystem::path &path, char (&dst)[Size]) {
        strncpy_s(dst, reinterpret_cast<const char*>(path.u8string().c_str()), Size - 1);
        dst[Size - 1] = '\0';
    }

    void genFilesText() {
        namespace fs = std::filesystem;

        std::string text;
        bool firstFile = true;
        for (int i = 0; i < m_entryInfos.size(); ++i) {
            const EntryInfo &entryInfo = m_entryInfos[i];
            if (!entryInfo.selected)
                continue;

            if (!firstFile)
                text += ",";
            text += entryInfo.entry.path().filename().string();
            firstFile = false;
        }

        strToChars(text, m_curFilesText);
    }

    void changeDirectory(const std::filesystem::path &dirPath) {
        namespace fs = std::filesystem;

        m_curDir = fs::canonical(dirPath);

        m_curDirBlocks.clear();
        for (auto it = m_curDir.begin(); it != m_curDir.end(); ++it) {
            std::string str = reinterpret_cast<const char*>(it->u8string().c_str());
            if (strcmp(str.c_str(), "\\") == 0) {
                continue;
            }
            m_curDirBlocks.push_back(str);
        }

        m_entryInfos.clear();
        EntryInfo upDirInfo;
        upDirInfo.entry = fs::directory_entry{ ".." };
        upDirInfo.selected = false;
        m_entryInfos.push_back(upDirInfo);
        m_numSelectedFiles = 0;
        m_numSelectedDirs = 0;
        for (const auto &entry : fs::directory_iterator(m_curDir)) {
            if ((m_flags & Flag_FileSelection) == 0 && !entry.is_directory())
                continue;
            EntryInfo info;
            info.entry = entry;
            info.selected = false;
            m_entryInfos.push_back(info);
        }

        pathToChars(m_curDir, m_curDirText);
        genFilesText();
    }

public:
    FileDialog() : m_curDir(""), m_font(nullptr) {
        m_title = "Default Title";
        m_flags = Flag_FileSelection;
    }

    void setFont(ImFont* font) {
        m_font = font;
    }

    void setFlags(uint32_t flags) {
        m_flags = static_cast<Flag>(flags);
        if (m_flags == 0)
            m_flags = Flag_FileSelection;
    }

    void show() {
        namespace fs = std::filesystem;

        changeDirectory(m_curDir.empty() ? fs::current_path() : m_curDir);
        ImGui::OpenPopup(m_title);
    }

    Result drawAndGetResult();

    void calcEntries(std::vector<std::filesystem::directory_entry>* entries);
};
