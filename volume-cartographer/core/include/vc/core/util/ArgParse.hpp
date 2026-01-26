#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vc::cli {

struct ParsedArgs {
    std::unordered_map<std::string, std::vector<std::string>> values;
    std::unordered_set<std::string> flags;
    std::vector<std::string> positionals;

    bool has(const std::string& name) const {
        return flags.contains(name) || values.contains(name);
    }

    std::string value(const std::string& name, const std::string& fallback = "") const {
        auto it = values.find(name);
        if (it == values.end() || it->second.empty())
            return fallback;
        return it->second.front();
    }
};

class ArgParser {
public:
    void add_option(const std::string& canonical,
                    const std::vector<std::string>& aliases,
                    bool required,
                    const std::string& help) {
        add_spec(canonical, aliases, true, required, help);
    }

    void add_flag(const std::string& canonical,
                  const std::vector<std::string>& aliases,
                  const std::string& help) {
        add_spec(canonical, aliases, false, false, help);
    }

    ParsedArgs parse(int argc, char** argv, std::string* error_out) const {
        ParsedArgs out;
        std::string error;
        bool collect_positionals = false;

        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            if (collect_positionals || token.empty() || token[0] != '-') {
                out.positionals.push_back(token);
                continue;
            }

            if (token == "--") {
                collect_positionals = true;
                continue;
            }

            bool has_value = false;
            std::string name;
            std::string value;

            if (token.rfind("--", 0) == 0) {
                name = token.substr(2);
            } else {
                name = token.substr(1);
            }

            auto eq = name.find('=');
            if (eq != std::string::npos) {
                value = name.substr(eq + 1);
                name = name.substr(0, eq);
                has_value = true;
            }

            auto it = alias_to_index_.find(name);
            if (it == alias_to_index_.end()) {
                error = "Unknown option: " + token;
                break;
            }

            const OptionSpec& spec = specs_[it->second];
            if (spec.requires_value) {
                if (!has_value) {
                    if (i + 1 >= argc) {
                        error = "Missing value for option: " + format_name(spec.canonical);
                        break;
                    }
                    value = argv[++i];
                }
                out.values[spec.canonical].push_back(value);
            } else {
                if (has_value) {
                    error = "Flag does not take a value: " + format_name(spec.canonical);
                    break;
                }
                out.flags.insert(spec.canonical);
            }
        }

        if (error.empty()) {
            for (const auto& spec : specs_) {
                if (!spec.required)
                    continue;
                if (!out.values.contains(spec.canonical)) {
                    error = "Missing required option: " + format_name(spec.canonical);
                    break;
                }
            }
        }

        if (error_out)
            *error_out = error;
        return out;
    }

    std::string help_text(const std::string& heading) const {
        std::ostringstream out;
        out << heading << "\n";
        for (const auto& spec : specs_) {
            out << "  " << format_names(spec.names);
            if (spec.requires_value)
                out << " <value>";
            if (spec.required)
                out << " (required)";
            if (!spec.help.empty())
                out << "  " << spec.help;
            out << "\n";
        }
        return out.str();
    }

private:
    struct OptionSpec {
        std::string canonical;
        std::vector<std::string> names;
        bool requires_value;
        bool required;
        std::string help;
    };

    static std::string format_name(const std::string& name) {
        if (name.size() == 1)
            return "-" + name;
        return "--" + name;
    }

    static std::string format_names(const std::vector<std::string>& names) {
        std::ostringstream out;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i != 0)
                out << ", ";
            out << format_name(names[i]);
        }
        return out.str();
    }

    void add_spec(const std::string& canonical,
                  const std::vector<std::string>& aliases,
                  bool requires_value,
                  bool required,
                  const std::string& help) {
        OptionSpec spec;
        spec.canonical = canonical;
        spec.requires_value = requires_value;
        spec.required = required;
        spec.help = help;
        spec.names.push_back(canonical);
        for (const auto& alias : aliases) {
            if (alias.empty())
                continue;
            spec.names.push_back(alias);
        }
        specs_.push_back(spec);
        size_t index = specs_.size() - 1;
        for (const auto& name : spec.names) {
            alias_to_index_[name] = index;
        }
    }

    std::vector<OptionSpec> specs_;
    std::unordered_map<std::string, size_t> alias_to_index_;
};

}  // namespace vc::cli
