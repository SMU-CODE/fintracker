from __future__ import annotations

import fnmatch
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import json5


# ============================================================================
# DOMAIN MODELS
# ============================================================================

@dataclass(frozen=True)
class PatternDefinition:
    """Named file pattern definition."""
    name: str
    pattern: str


@dataclass(frozen=True)
class ProjectConfig:
    """Project configuration settings."""
    
    base_project_folder: str = "."
    output_collection_path: str = "output/collected_content.txt"
    output_tree_path: str = "output/ProjectTree.txt"
    ignore_patterns: List[str] = field(default_factory=list)
    folder_priority: List[str] = field(default_factory=list)
    file_patterns: List[Any] = field(default_factory=list)  # Can be both PatternDefinition and str
    skip_todo_comments: bool = True
    remove_singleline_comment: bool = True
    remove_multiline_comments: bool = True
    enable_output_header: bool = False
    ignore_hidden_items: bool = True
    scan_subfolders: bool = True
    add_tree_spacing: bool = False
    max_worker_threads: int = min(32, (os.cpu_count() or 4) + 4)
    use_gitignore: bool = True
    project_type: str = "general"
    allowed_extensions: List[str] = field(default_factory=list)
    show_empty_folders: bool = False
    lock_configuration: bool = True
    reset_on_next_load: bool = False
    apply_core_settings: bool = True
    linear_display_style_enable: bool = True
    enable_export_as_separate_files: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectConfig:
        """Create config from dictionary with hybrid patterns."""
        processed_data = data.copy()
        
        if 'file_patterns' in processed_data:
            processed_patterns = []
            for item in processed_data['file_patterns']:
                if isinstance(item, dict):
                    processed_patterns.append(
                        PatternDefinition(name=item['name'], pattern=item['pattern'])
                    )
                elif isinstance(item, str):
                    processed_patterns.append(
                        PatternDefinition(name=item, pattern=item)
                    )
            processed_data['file_patterns'] = processed_patterns
        
        return cls(**{k: processed_data[k] for k in processed_data if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary preserving hybrid structure."""
        result = asdict(self)
        
        if self.file_patterns and isinstance(self.file_patterns[0], PatternDefinition):
            restored_patterns = []
            for pattern_def in self.file_patterns:
                if pattern_def.name == pattern_def.pattern:
                    restored_patterns.append(pattern_def.pattern)
                else:
                    restored_patterns.append({
                        'name': pattern_def.name,
                        'pattern': pattern_def.pattern
                    })
            result['file_patterns'] = restored_patterns
        
        return result


@dataclass(frozen=True)
class FileInfo:
    """File information with cleaned content."""
    path: Path
    relative_path: str
    content: str


@dataclass(frozen=True)
class CollectionResult:
    """Result of file collection operation."""
    total_files: int
    content: str
    info: str
    elapsed_time: float


class CollectionDepth(Enum):
    """Collection depth options."""
    SHALLOW = "shallow"
    DEEP = "deep"


class ProjectType(Enum):
    """Supported project types."""
    GENERAL = "general"
    PYTHON = "python"
    DJANGO = "django"
    FLUTTER = "flutter"
    LARAVEL = "laravel"
    NODEJS = "nodejs"
    ANGULAR = "angular"
    DOTNET = "dotnet"
    PHP = "php"


# ============================================================================
# PROTOCOLS (Interfaces)
# ============================================================================

@runtime_checkable
class ConfigLoader(Protocol):
    def load(self) -> ProjectConfig: ...


@runtime_checkable
class ConfigSaver(Protocol):
    def save(self, config: ProjectConfig) -> None: ...


@runtime_checkable
class ConfigResetter(Protocol):
    def reset(self) -> ProjectConfig: ...


@runtime_checkable
class ConfigProvider(ConfigLoader, ConfigSaver, ConfigResetter, Protocol):
    pass


@runtime_checkable
class TreeGenerator(Protocol):
    def generate_file(self, root: Path, include_files: bool = True) -> bool: ...
    def generate_interactive_map(self, root: Path) -> Dict[int, Dict[str, Any]]: ...


@runtime_checkable
class FileProcessorProtocol(Protocol):
    def process(self, path: Path) -> Optional[FileInfo]: ...


@runtime_checkable
class FileDiscovererProtocol(Protocol):
    def find(self, root: Path, depth: CollectionDepth) -> List[Path]: ...
    def find_by_patterns(self, root: Path, patterns: List[str]) -> List[Path]: ...
    def is_ignored(self, path: Path) -> bool: ...


@runtime_checkable
class FileCollector(Protocol):
    def collect_from_folders(self, folders: List[Dict[str, Any]]) -> Tuple[int, str]: ...
    def collect_by_patterns(self, patterns: List[str]) -> Tuple[int, str]: ...


@runtime_checkable
class OutputWriter(Protocol):
    def write(self, result: CollectionResult) -> None: ...


@runtime_checkable
class ProgressListener(Protocol):
    def on_start(self, total: int) -> None: ...
    def on_update(self, processed: int, total: int) -> None: ...
    def on_complete(self, elapsed_time: float) -> None: ...


@runtime_checkable
class SelectionParser(Protocol):
    def parse_folder_selection(self, sel_str: str, f_map: Dict) -> List[Dict[str, Any]]: ...
    def parse_pattern_selection(self, sel_str: str, file_patterns: List[PatternDefinition]) -> List[str]: ...


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class JsonConfigProvider:
    """Handles configuration I/O using JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

    def load(self) -> ProjectConfig:
        if not self.path.exists():
            return self._create_default()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json5.load(f)
            config = ProjectConfig.from_dict(data)

            if config.reset_on_next_load:
                print("🔄 Settings reset requested. Resetting to defaults...")
                return self.reset()

            print(f"✅ Config loaded from: {self.path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}. Creating default.")
            return self._create_default()

    def save(self, config: ProjectConfig) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as f:
                json5.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\n✅ Config saved to: {self.path}")
        except IOError as e:
            print(f"❌ Error saving config: {e}")

    def reset(self) -> ProjectConfig:
        """Reset to factory defaults."""
        current_project_type = ProjectTypeDetector.detect(Path.cwd())
        print(f"🔄 Resetting config for project type: {current_project_type.value}")

        default_config = ProjectConfigFactory.create(current_project_type)
        self.save(default_config)
        print(f"🔄 Config reset to {current_project_type.value} defaults.")
        return default_config

    def _create_default(self) -> ProjectConfig:
        """Create and save default config."""
        current_project_type = ProjectTypeDetector.detect(Path.cwd())
        default_config = ProjectConfigFactory.create(current_project_type)
        self.save(default_config)
        print(f"🆕 Created {current_project_type.value} config file.")
        return default_config


class ProjectTypeRegistry:
    """Dynamic registry for project types."""

    _project_types: Dict[ProjectType, Dict[str, Any]] = {}

    @classmethod
    def register(cls, project_type: ProjectType, settings: Dict[str, Any]):
        cls._project_types[project_type] = settings

    @classmethod
    def get_settings(cls, project_type: ProjectType) -> Dict[str, Any]:
        return cls._project_types.get(project_type, cls._project_types[ProjectType.GENERAL])

    @classmethod
    def initialize_defaults(cls):
        """Initialize default settings for all project types."""
        defaults = {
            ProjectType.PYTHON: {
                "project_type": "python",
                "allowed_extensions": [".py", ".json", ".yaml", ".txt", ".md"],
                "file_patterns": [
                    # Individual patterns
                    "*.py",
                    "requirements.txt",
                    "*_model.py",
                    "main.py",
                    "app.py",
                    # Grouped patterns
                    {
                        "name": "Python: All Important Files",
                        "pattern": "*.py,requirements.txt,setup.py,README.md"
                    },
                    {
                        "name": "Python: Models & Main Files",
                        "pattern": "*_model.py,main.py,app.py"
                    }
                ],
                "folder_priority": ["src", "app", "core", "utils"],
            },
            ProjectType.DJANGO: {
                "project_type": "django",
                "allowed_extensions": [".py", ".html", ".css", ".js", ".json", ".sql", ".md"],
                "file_patterns": [
                    # Individual patterns
                    "models.py",
                    "admin.py",
                    "*views.py",
                    "urls.py",
                    "settings.py",
                    "manage.py",
                    # Grouped patterns
                    {
                        "name": "Django: Core Files",
                        "pattern": "models.py,views.py,urls.py,admin.py"
                    },
                    {
                        "name": "Django: Config & Management",
                        "pattern": "settings.py,manage.py,urls.py"
                    }
                ],
                "folder_priority": ["manage.py", "settings", "static", "templates", "core", "utils"],
                "ignore_patterns": ["migrations", "staticfiles", "media", "db.sqlite3"],
            },
            ProjectType.FLUTTER: {
                "project_type": "flutter",
                "allowed_extensions": [".dart", ".yaml", ".json", ".md", ".arb"],
                "file_patterns": [
                    # Individual patterns
                    "*_model.dart",
                    "*_repository.dart",
                    "*_providers.dart",
                    "*_notifier.dart",
                    "di_*.dart",
                    "*_screen.dart",
                    "*_widget.dart",
                    "*_dialog.dart",
                    "features/**/*_model.dart",
                    "features/**/*_repository.dart",
                    # Grouped patterns
                    {
                        "name": "Flutter: All Core Logic",
                        "pattern": "*_model.dart,*_repository.dart,*_providers.dart,*_notifier.dart,di_*.dart"
                    },
                    {
                        "name": "Flutter: All UI Components",
                        "pattern": "*_screen.dart,*_widget.dart,*_dialog.dart"
                    }
                ],
                "folder_priority": ["lib", "assets"],
                "ignore_patterns": [".dart_tool", "build", "*.yaml"],
            },
            ProjectType.LARAVEL: {
                "project_type": "laravel",
                "allowed_extensions": [".php", ".blade.php", ".js", ".css", ".json"],
                "file_patterns": [
                    # Individual patterns
                    "*Controller.php",
                    "*.php",
                    "*.blade.php",
                    # Grouped patterns
                    {
                        "name": "Laravel: MVC Files",
                        "pattern": "*Controller.php,*.php,*.blade.php"
                    },
                    {
                        "name": "Laravel: Config Files",
                        "pattern": "*.php,*.json"
                    }
                ],
                "folder_priority": ["app", "routes", "resources", "config", "database"],
                "ignore_patterns": ["storage", "bootstrap/cache"],
            },
            ProjectType.NODEJS: {
                "project_type": "nodejs",
                "allowed_extensions": [".js", ".ts", ".json", ".html", ".css", ".md"],
                "file_patterns": [
                    # Individual patterns
                    "package.json",
                    "index.js",
                    "server.js",
                    "app.js",
                    # Grouped patterns
                    {
                        "name": "Node.js: Config & Entry Points",
                        "pattern": "package.json,index.js,server.js,app.js"
                    },
                    {
                        "name": "Node.js: All JS/TS Files",
                        "pattern": "*.js,*.ts"
                    }
                ],
                "folder_priority": ["src", "lib", "routes", "middleware", "config", "public"],
            },
            ProjectType.ANGULAR: {
                "project_type": "angular",
                "allowed_extensions": [".ts", ".html", ".css", ".scss", ".json"],
                "file_patterns": [
                    # Individual patterns
                    "package.json",
                    "angular.json",
                    "*.module.ts",
                    "*.component.ts",
                    # Grouped patterns
                    {
                        "name": "Angular: Config Files",
                        "pattern": "package.json,angular.json"
                    },
                    {
                        "name": "Angular: Component Files",
                        "pattern": "*.module.ts,*.component.ts"
                    }
                ],
                "folder_priority": ["src", "app", "assets", "environments", "core"],
            },
            ProjectType.DOTNET: {
                "project_type": "dotnet",
                "allowed_extensions": [".cs", ".cshtml", ".json", ".csproj", ".sln"],
                "file_patterns": [
                    # Individual patterns
                    "*.csproj",
                    "*.sln",
                    "appsettings.json",
                    "Program.cs",
                    "Startup.cs",
                    # Grouped patterns
                    {
                        "name": ".NET: Project Files",
                        "pattern": "*.csproj,*.sln"
                    },
                    {
                        "name": ".NET: Core Files",
                        "pattern": "Program.cs,Startup.cs,appsettings.json"
                    }
                ],
                "folder_priority": ["Controllers", "Models", "Views", "Services", "Data", "wwwroot"],
            },
            ProjectType.PHP: {
                "project_type": "php",
                "allowed_extensions": [".php", ".html", ".css", ".js", ".json", ".xml", ".sql", ".md"],
                "file_patterns": [
                    # Individual patterns
                    "*.php",
                    "index.php",
                    "composer.json",
                    # Grouped patterns
                    {
                        "name": "PHP: All PHP Files",
                        "pattern": "*.php,index.php"
                    },
                    {
                        "name": "PHP: Config Files",
                        "pattern": "composer.json,*.json"
                    }
                ],
                "folder_priority": ["src", "app", "config", "public", "views", "models", "controllers"],
            },
            ProjectType.GENERAL: {
                "project_type": "general",
                "allowed_extensions": [".py", ".js", ".html", ".css", ".json", ".md", ".txt"],
                "file_patterns": [
                    # Individual patterns
                    "README.md",
                    "requirements.txt",
                    "package.json",
                    # Grouped patterns
                    {
                        "name": "General: Documentation",
                        "pattern": "README.md,*.txt,*.md"
                    },
                    {
                        "name": "General: Config Files",
                        "pattern": "requirements.txt,package.json,*.json"
                    }
                ],
                "folder_priority": ["src", "lib", "app", "config", "docs", "tests"],
            },
        }
        for ptype, settings in defaults.items():
            cls.register(ptype, settings)


class ProjectConfigFactory:
    """Creates project-specific configurations."""

    CORE_SETTINGS = {
        "base_project_folder": ".",
        "ignore_patterns": [
            ".*", "*.pyc", "*.pyo", "*.pyd", "*.log", "*.user", "*.suo", "*.egg-info",
            "__pycache__", ".pytest_cache", ".mypy_cache", ".coverage", "htmlcov",
            "venv", "env", ".env", "dist", "build", "target", "node_modules", ".git",
            "output", "bin", "obj", "packages", "vendor", "test", "helper prompts.txtz_test",
        ],
        "folder_priority": ["shared", "main.py", "app.py", "requirements.txt", "setup.py"],
    }

    @staticmethod
    def create(project_type: ProjectType) -> ProjectConfig:
        """Create complete config for given project type."""
        if not ProjectTypeRegistry._project_types:
            ProjectTypeRegistry.initialize_defaults()

        specific_settings = ProjectTypeRegistry.get_settings(project_type)
        final_settings_dict = ProjectConfigFactory._apply_core_settings(specific_settings)

        return ProjectConfig.from_dict(final_settings_dict)

    @staticmethod
    def _apply_core_settings(specific: Dict[str, Any]) -> Dict[str, Any]:
        """Merge specific settings with core settings."""
        merged_ignore = list(
            dict.fromkeys(specific.get("ignore_patterns", []) + ProjectConfigFactory.CORE_SETTINGS["ignore_patterns"])
        )
        merged_priority = list(
            dict.fromkeys(ProjectConfigFactory.CORE_SETTINGS["folder_priority"] + specific.get("folder_priority", []))
        )

        final_config = specific.copy()
        final_config["ignore_patterns"] = merged_ignore
        final_config["folder_priority"] = merged_priority
        final_config["base_project_folder"] = specific.get("base_project_folder", ".")
        final_config["apply_core_settings"] = True
        return final_config


class ProjectTypeDetector:
    """Detects project type based on characteristic files."""

    @staticmethod
    def detect(root: Path) -> ProjectType:
        """Heuristically detect project type."""
        if (root / "manage.py").exists():
            return ProjectType.DJANGO
        if (root / "pubspec.yaml").exists():
            return ProjectType.FLUTTER
        if (root / "artisan").exists():
            return ProjectType.LARAVEL
        if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
            return ProjectType.DOTNET

        package_json = root / "package.json"
        if package_json.exists():
            try:
                data = json5.loads(package_json.read_text(encoding="utf-8"))
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                if "@angular/core" in deps:
                    return ProjectType.ANGULAR
            except (json5.JSONDecodeError, UnicodeDecodeError):
                pass
            return ProjectType.NODEJS

        if (root / "composer.json").exists():
            return ProjectType.PHP
        if len(list(root.glob("**/*.py"))) > 2:
            return ProjectType.PYTHON

        return ProjectType.GENERAL


class ConfigManager:
    """Manages application configuration."""

    @staticmethod
    def create_final_config(config_provider: ConfigProvider) -> ProjectConfig:
        current_config = config_provider.load()

        if current_config.lock_configuration and current_config.project_type != "general":
            print("🔒 Configuration is locked. Using saved settings.")
            return current_config
        else:
            detected_type = ProjectTypeDetector.detect(Path.cwd())
            print(f"🔍 Detected project type: {detected_type.value}")
            final_config = ProjectConfigFactory.create(detected_type)
            config_provider.save(final_config)
            return final_config


# ============================================================================
# FILE PROCESSING
# ============================================================================

class CommentRemover:
    """Remove comments from code content."""

    def remove(self, content: str, ext: str, config: ProjectConfig) -> str:
        if not config.remove_multiline_comments and not config.remove_singleline_comment:
            return content

        if config.remove_multiline_comments:
            content = self._remove_multiline_comments(content, ext)

        if config.remove_singleline_comment:
            content = self._remove_singleline_comments(content, ext, config)

        return content

    def _remove_multiline_comments(self, content: str, ext: str) -> str:
        patterns = [(r"/\*.*?\*/", re.DOTALL)]

        if ext == ".py":
            patterns.extend([
                (r'"""(.*?)"""', re.DOTALL),
                (r"'''(.*?)'''", re.DOTALL),
            ])
        elif ext == ".dart":
            patterns.append((r'/\*\*.*?\*/', re.DOTALL))

        for pattern, flags in patterns:
            content = re.sub(pattern, "", content, flags=flags)

        return content

    def _remove_singleline_comments(self, content: str, ext: str, config: ProjectConfig) -> str:
        if ext == ".dart":
            lines = content.split("\n")
            cleaned_lines = []
            
            for line in lines:
                comment_pos = -1
                pos_triple_slash = line.find("///")
                pos_double_slash = line.find("//")
                
                if pos_triple_slash != -1:
                    comment_pos = pos_triple_slash
                
                if pos_double_slash != -1:
                    if comment_pos == -1 or pos_double_slash < comment_pos:
                        comment_pos = pos_double_slash
                
                if comment_pos != -1:
                    if config.skip_todo_comments and self._is_todo_comment(line[comment_pos:].lower()):
                        cleaned_lines.append(line)
                    else:
                        cleaned_line = line[:comment_pos].rstrip()
                        cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)

            return "\n".join(line for line in cleaned_lines if line.strip())
        
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            comment_marker = self._get_comment_marker(ext, line)

            if comment_marker and comment_marker in line:
                comment_pos = line.find(comment_marker)

                if config.skip_todo_comments and self._is_todo_comment(line[comment_pos:].lower()):
                    cleaned_lines.append(line)
                else:
                    cleaned_line = line[:comment_pos].rstrip()
                    cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)

        return "\n".join(line for line in cleaned_lines if line.strip())

    def _get_comment_marker(self, ext: str, line: str = "") -> Optional[str]:
        ext = ext.lower()

        if ext in [".py", ".rb", ".yaml", ".yml", ".sh", ".bash", ".zsh", ".dockerfile", ".pl", ".pm", ".r", ".cfg", ".ini", ".toml"]:
            return "#"
        elif ext in [".js", ".ts", ".jsx", ".tsx", ".java", ".cs", ".cpp", ".c", ".m", ".mm", ".php", ".swift", ".go", ".rs", ".kt", ".scala"]:
            return "//"
        elif ext == ".dart":
            return "//"
        elif ext in [".html", ".htm", ".xml", ".vue", ".svelte", ".astro"]:
            if "<!--" in line:
                return "<!--"
            return None
        elif ext in [".css", ".scss", ".sass", ".less"]:
            if ext in [".scss", ".sass", ".less"] and "//" in line:
                return "//"
            return None
        elif ext in [".sql", ".psql"]:
            return "--"
        elif ext == ".lua":
            return "--"
        elif ext in [".m", ".matlab"]:
            return "%"

        return None

    def _is_todo_comment(self, comment_text: str) -> bool:
        todo_indicators = ["todo", "fixme", "xxx", "hack", "bug", "note"]
        comment_lower = comment_text.lower()
        return any(indicator in comment_lower for indicator in todo_indicators)


class FileProcessor:
    """Reads, cleans, and packages a single file's info."""

    def __init__(self, config: ProjectConfig, remover: CommentRemover):
        self.config = config
        self.remover = remover

    def process(self, path: Path) -> Optional[FileInfo]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            cleaned_content = self.remover.remove(content, path.suffix.lower(), self.config)

            try:
                relative_path = path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                relative_path = path.absolute().as_posix()

            return FileInfo(path=path, relative_path=relative_path, content=cleaned_content)
        except Exception as e:
            print(f"⚠️ Could not process {path.name}: {e}")
            return None


class IgnorePatternLoader:
    """Handles loading ignore patterns."""

    def __init__(self, config: ProjectConfig, root_path: Path = None):
        self.config = config
        self.root_path = root_path if root_path else Path.cwd()
        self.patterns = self._load_all_patterns()
        self._cache: Dict[Path, bool] = {}

    def _load_all_patterns(self) -> List[str]:
        patterns = self.config.ignore_patterns.copy()

        if self.config.ignore_hidden_items:
            patterns.extend([".*", "*/.*"])

        if self.config.use_gitignore:
            patterns.extend(self._load_gitignore_patterns(self.root_path))

        return list(dict.fromkeys(patterns))

    def _load_gitignore_patterns(self, root_path: Path) -> List[str]:
        patterns = []
        gitignore_path = root_path / ".gitignore"

        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if line.startswith("/"):
                            line = line[1:]
                        if line.endswith("/"):
                            patterns.append(line.rstrip("/"))
                            patterns.append(f"{line}*")
                        else:
                            patterns.append(line)
            except Exception as e:
                print(f"⚠️ Could not read .gitignore: {e}")

        return patterns

    def is_ignored(self, path: Path) -> bool:
        if path in self._cache:
            return self._cache[path]

        result = self._check_ignored(path)
        self._cache[path] = result
        return result

    def _check_ignored(self, path: Path) -> bool:
        path_str = path.as_posix()

        for pattern in self.patterns:
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                if (
                    fnmatch.fnmatch(path_str, dir_pattern)
                    or fnmatch.fnmatch(path.name, dir_pattern)
                    or (path.is_dir() and fnmatch.fnmatch(path_str, f"*/{dir_pattern}/*"))
                ):
                    return True
            elif fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True

        return False


class FileDiscoverer:
    """Discovers files and applies ignore/extension rules."""

    def __init__(self, config: ProjectConfig, pattern_loader: IgnorePatternLoader):
        self.config = config
        self.pattern_loader = pattern_loader

    def is_ignored(self, path: Path) -> bool:
        return self.pattern_loader.is_ignored(path)

    def find(self, root: Path, depth: CollectionDepth) -> List[Path]:
        search_root = root / self.config.base_project_folder
        if not search_root.exists() or not search_root.is_dir():
            search_root = root

        if self.is_ignored(search_root):
            return []

        files = []
        if depth == CollectionDepth.SHALLOW:
            for item in search_root.iterdir():
                if item.is_file() and not self.is_ignored(item) and self._is_allowed_ext(item):
                    files.append(item)
        else:
            for dirpath, _, filenames in os.walk(search_root):
                d_path = Path(dirpath)
                if self.is_ignored(d_path):
                    continue
                for f in filenames:
                    f_path = d_path / f
                    if not self.is_ignored(f_path) and self._is_allowed_ext(f_path):
                        files.append(f_path)
        return files

    def find_by_patterns(self, root: Path, patterns: List[str]) -> List[Path]:
        """Find files matching given patterns."""
        search_root = root / self.config.base_project_folder
        if not search_root.exists() or not search_root.is_dir():
            search_root = root

        expanded_patterns = []
        for pattern in patterns:
            if "," in pattern:
                sub_patterns = [p.strip() for p in pattern.split(",") if p.strip()]
                expanded_patterns.extend(sub_patterns)
            else:
                expanded_patterns.append(pattern)
        
        print(f"🔍 Searching for {len(expanded_patterns)} patterns in {search_root}")
        files = set()

        for pattern in expanded_patterns:
            try:
                if "**" in pattern:
                    for file_path in search_root.rglob(pattern.replace("**", "*")):
                        if (
                            file_path.is_file()
                            and not self.is_ignored(file_path)
                            and self._is_allowed_ext(file_path)
                        ):
                            files.add(file_path)
                else:
                    for file_path in search_root.glob(pattern):
                        if (
                            file_path.is_file()
                            and not self.is_ignored(file_path)
                            and self._is_allowed_ext(file_path)
                        ):
                            files.add(file_path)
            except Exception as e:
                print(f"⚠️ Pattern '{pattern}' failed: {e}")

        for dirpath, _, filenames in os.walk(search_root):
            d_path = Path(dirpath)
            if self.is_ignored(d_path):
                continue

            for f in filenames:
                f_path = d_path / f
                if self.is_ignored(f_path) or not self._is_allowed_ext(f_path):
                    continue

                try:
                    rel_path = f_path.relative_to(search_root).as_posix()
                except Exception:
                    continue

                for pattern in expanded_patterns:
                    if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(rel_path, pattern):
                        files.add(f_path)
                        break

        unique_files = sorted(list(files), key=lambda x: str(x))
        print(f"✅ Found {len(unique_files)} unique files matching patterns")

        if unique_files:
            print("📄 First 5 matching files:")
            for f in unique_files[:5]:
                try:
                    rel_path = f.relative_to(search_root)
                    print(f"  - {rel_path}")
                except Exception:
                    print(f"  - {f.name}")
        else:
            print("❌ No files found matching the patterns")

        return unique_files

    def _is_allowed_ext(self, path: Path) -> bool:
        return not self.config.allowed_extensions or path.suffix.lower() in self.config.allowed_extensions


class CommentFormatter:
    """Format comments for different file types."""

    @staticmethod
    def get_comment_prefix(file_path: Path) -> str:
        ext = file_path.suffix.lower()

        if ext in [".dart", ".js", ".ts", ".java", ".cs", ".cpp", ".c", ".php", ".swift", ".go"]:
            return "//"
        elif ext in [".py", ".rb", ".sh", ".yaml", ".yml", ".dockerfile", ""]:
            return "#"
        elif ext in [".html", ".xml", ".svg"]:
            return "<!--"
        elif ext in [".sql"]:
            return "--"
        elif ext in [".css"]:
            return "/*"
        else:
            return "//"

    @staticmethod
    def format_comment(text: str, comment_prefix: str) -> str:
        if comment_prefix == "<!--":
            return f"{comment_prefix} {text} -->"
        elif comment_prefix == "/*":
            return f"{comment_prefix} {text} */"
        else:
            return f"{comment_prefix} {text}"


class ContentOrganizer:
    """Organize collected content."""

    def __init__(self, formatter: CommentFormatter):
        self.formatter = formatter

    def organize_folder_content(self, folder: Dict[str, Any], folder_files: List[FileInfo]) -> str:
        if not folder_files:
            return ""

        comment_prefix = self.formatter.get_comment_prefix(folder_files[0].path) if folder_files else "//"
        depth_label = "DEEP" if folder["depth"] == CollectionDepth.DEEP else "SHALLOW"
        folder_comment = self.formatter.format_comment(
            f"=== FOLDER: {folder['path'].name} ({depth_label}) ===", comment_prefix
        )

        content = f"\n{folder_comment}\n"
        for info in sorted(folder_files, key=lambda i: i.relative_path):
            file_comment_prefix = self.formatter.get_comment_prefix(info.path)
            file_comment = self.formatter.format_comment(f"--- FILE: {info.relative_path} ---", file_comment_prefix)
            content += f"\n{file_comment}\n{info.content}\n"

        return content

    def organize_pattern_content(self, pattern: str, pattern_files: List[FileInfo]) -> str:
        if not pattern_files:
            return ""

        comment_prefix = self.formatter.get_comment_prefix(pattern_files[0].path) if pattern_files else "//"
        pattern_comment = self.formatter.format_comment(
            f"=== PATTERN: {pattern} ({len(pattern_files)} files) ===", comment_prefix
        )

        content = f"\n{pattern_comment}\n"
        for info in sorted(pattern_files, key=lambda i: i.relative_path):
            file_comment_prefix = self.formatter.get_comment_prefix(info.path)
            file_comment = self.formatter.format_comment(f"--- FILE: {info.relative_path} ---", file_comment_prefix)
            content += f"\n{file_comment}\n{info.content}\n"

        return content


class ParallelFileCollector:
    """Collects and processes files in parallel."""

    def __init__(
        self,
        processor: FileProcessor,
        discoverer: FileDiscoverer,
        listener: ProgressListener,
        organizer: ContentOrganizer,
    ):
        self.processor = processor
        self.discoverer = discoverer
        self.listener = listener
        self.organizer = organizer

    def collect_from_folders(self, folders: List[Dict[str, Any]]) -> Tuple[int, str]:
        all_files = []
        for folder in folders:
            depth = folder["depth"]
            all_files.extend(self.discoverer.find(folder["path"], depth))

        files_to_process = sorted(list(set(all_files)))
        if not files_to_process:
            return 0, ""

        processed_infos = self._process_in_parallel(files_to_process)

        content = ""
        for folder in folders:
            folder_files = [info for info in processed_infos if info.path.is_relative_to(folder["path"])]
            if folder_files:
                content += self.organizer.organize_folder_content(folder, folder_files)

        return len(processed_infos), content

    def collect_by_patterns(self, patterns: List[str]) -> Tuple[int, str]:
        print(f"🚀 Starting search for patterns: {patterns}")
        files_to_process = self.discoverer.find_by_patterns(Path.cwd(), patterns)

        if not files_to_process:
            print("❌ No files found matching patterns")
            return 0, ""

        print(f"📁 Found {len(files_to_process)} files to process...")
        processed_infos = self._process_in_parallel(files_to_process)
        print(f"📊 Total processed info objects: {len(processed_infos)}")

        content = ""
        if processed_infos:
            for pattern in patterns:
                pattern_files = []
                for info in processed_infos:
                    if fnmatch.fnmatch(info.path.name, pattern) or fnmatch.fnmatch(
                        info.relative_path, pattern
                    ):
                        pattern_files.append(info)

                print(f"\n  Pattern '{pattern}' matched {len(pattern_files)} files")

                if pattern_files:
                    content += self.organizer.organize_pattern_content(pattern, pattern_files)

            if not content.strip():
                print("⚠️ No content after organization, creating simple format...")
                comment_prefix = CommentFormatter.get_comment_prefix(processed_infos[0].path)
                pattern_comment = CommentFormatter.format_comment(
                    f"=== PATTERNS: {', '.join(patterns)} ({len(processed_infos)} files) ===",
                    comment_prefix,
                )

                content = f"\n{pattern_comment}\n"
                for info in sorted(processed_infos, key=lambda i: i.relative_path):
                    file_comment = CommentFormatter.format_comment(
                        f"--- FILE: {info.relative_path} ---", comment_prefix
                    )
                    content += f"\n{file_comment}\n{info.content}\n"

        return len(processed_infos), content

    def _process_in_parallel(self, files: List[Path]) -> List[FileInfo]:
        self.listener.on_start(len(files))
        start_time = time.monotonic()

        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(self.processor.config.max_worker_threads) as executor:
            future_to_path = {executor.submit(self.processor.process, path): path for path in files}
            for i, future in enumerate(as_completed(future_to_path)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        successful += 1
                    else:
                        failed += 1
                        path = future_to_path[future]
                        print(f"  ✗ Failed to process: {path.name}")
                except Exception as e:
                    failed += 1
                    path = future_to_path[future]
                    print(f"  ✗ Error processing {path.name}: {e}")

                self.listener.on_update(i + 1, len(files))

        self.listener.on_complete(time.monotonic() - start_time)
        print(f"📊 Processing summary: {successful} successful, {failed} failed")
        return results


# ============================================================================
# TREE GENERATION
# ============================================================================

class EnhancedTreeGenerator:
    """Generate textual representation of directory structure."""

    def __init__(self, config: ProjectConfig, discoverer: FileDiscoverer):
        self.config = config
        self.discoverer = discoverer

    def _has_valid_content(self, path: Path) -> bool:
        """Check if directory contains any valid files."""
        try:
            for item in path.iterdir():
                if self.discoverer.is_ignored(item):
                    continue
                if item.is_file() and self.discoverer._is_allowed_ext(item):
                    return True
                if item.is_dir() and self._has_valid_content(item):
                    return True
            return False
        except (PermissionError, FileNotFoundError):
            return False

    def _sort_entries(self, entries: List[Path]) -> List[Path]:
        """Sort entries with files first, then folders with priority."""
        files = [e for e in entries if e.is_file()]
        dirs = [e for e in entries if e.is_dir()]

        files_sorted = sorted(files, key=lambda p: p.name.lower())
        priority = self.config.folder_priority

        def get_folder_sort_key(path: Path) -> tuple:
            if path.name in priority:
                return (priority.index(path.name), path.name.lower())
            return (len(priority), path.name.lower())

        dirs_sorted = sorted(dirs, key=get_folder_sort_key)
        return files_sorted + dirs_sorted

    def generate_file(self, root: Path, include_files: bool = True) -> bool:
        """Generate and save directory tree to file."""
        if self.config.enable_export_as_separate_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path(self.config.output_tree_path)

            if base_path.suffix:
                new_filename = f"{base_path.stem}_{timestamp}{base_path.suffix}"
            else:
                new_filename = f"{base_path.name}_{timestamp}"

            output_path = base_path.parent / new_filename
        else:
            output_path = Path(self.config.output_tree_path)

        output_path.parent.mkdir(exist_ok=True)
        try:
            base_folder_path = root / self.config.base_project_folder
            if not base_folder_path.exists() or not base_folder_path.is_dir():
                base_folder_path = root

            tree_type = "Full Structure" if include_files else "Folders Only"
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"Project Tree for: {base_folder_path} ({tree_type})\n" + "=" * 50 + "\n")
                self._write_tree_recursive(base_folder_path, f, include_files)
            print(f"✅ Tree saved to {output_path} ({tree_type})")
            return True
        except IOError as e:
            print(f"❌ Error writing tree file: {e}")
            return False

    def _write_tree_recursive(self, current_path: Path, file, include_files: bool, prefix: str = ""):
        """Helper to recursively write tree structure."""
        try:
            entries = self._sort_entries([p for p in current_path.iterdir() if not self.discoverer.is_ignored(p)])
        except (FileNotFoundError, PermissionError):
            return

        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file() and self.discoverer._is_allowed_ext(e)] if include_files else []

        all_items = files + dirs

        for i, item in enumerate(all_items):
            is_last = i == len(all_items) - 1

            if item.is_dir() and not self.config.show_empty_folders and not self._has_valid_content(item):
                continue

            connector = "└── " if is_last else "├── "
            spacing = "\n" if self.config.add_tree_spacing else ""
            item_display = f"{item.name}" if item.is_file() else item.name

            file.write(f"{prefix}{connector}{item_display}{spacing}\n")

            if item.is_dir() and self.config.scan_subfolders:
                new_prefix = prefix + ("    " if is_last else "│   ")
                self._write_tree_recursive(item, file, include_files, new_prefix)

    def generate_interactive_map(self, root: Path) -> Dict[int, Dict[str, Any]]:
        """Generate numbered map of directories for interactive selection."""
        print("\n" + "─" * 60)
        print("📁 Enhanced Folder Structure:")
        print("// Enter number for SHALLOW scan (e.g., 3)")
        print("// Enter *number for DEEP scan (e.g., *3)")
        print("// Multiple selections: 1, *3, 5")
        print("=" * 60)

        base_folder_path = root / self.config.base_project_folder
        if not base_folder_path.exists() or not base_folder_path.is_dir():
            print(f"⚠️ Base project folder '{self.config.base_project_folder}' not found, using root")
            base_folder_path = root

        folder_map = {
            0: {
                "path": base_folder_path,
                "name": base_folder_path.name,
                "type": "base",
                "depth": CollectionDepth.DEEP,
            }
        }

        if self.config.linear_display_style_enable:
            print(f"[0]-{base_folder_path.name} >> [ base project Folder ] <<")
        else:
            print(f"└──[0]---{base_folder_path.name} >> base project Folder <<")

        self._display_recursive(base_folder_path, folder_map, 1, 1, base_folder_path.name)
        print("─" * 60)
        return folder_map

    def _display_recursive(self, path: Path, f_map: Dict, index: int, depth: int, parent_path: str) -> int:
        """Helper to recursively display interactive tree."""
        try:
            dirs = self._sort_entries([p for p in path.iterdir() if p.is_dir() and not self.discoverer.is_ignored(p)])
        except (FileNotFoundError, PermissionError):
            return index

        for i, d in enumerate(dirs):
            if not self.config.show_empty_folders and not self._has_valid_content(d):
                continue

            is_root_level = depth == 1
            is_last = i == len(dirs) - 1

            f_map[index] = {
                "path": d,
                "name": d.name,
                "type": "root" if is_root_level else "sub",
                "depth": CollectionDepth.DEEP,
            }

            if self.config.linear_display_style_enable:
                full_path = f"{parent_path}/{d.name}"
                print(f"[{index}]-{full_path}{f' >> [root] >> ------[{index}]' if is_root_level else ''}")
            else:
                indent = "│  " * (depth - 1)
                connector = "└──" if is_last else "├──"
                print(f"{indent}{connector}[{index}]---{d.name}{' >> root' if is_root_level else ''}")

            index += 1
            if self.config.scan_subfolders:
                next_parent_path = f"{parent_path}/{d.name}" if self.config.linear_display_style_enable else ""
                index = self._display_recursive(d, f_map, index, depth + 1, next_parent_path)
        return index


# ============================================================================
# OUTPUT & PROGRESS
# ============================================================================

class FileOutputWriter:
    """Write collected content to file."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def write(self, result: CollectionResult) -> None:
        if not result.content or not result.content.strip():
            print(f"⚠️ No content to write. Total files: {result.total_files}")
            return

        if self.config.enable_export_as_separate_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path(self.config.output_collection_path)

            if base_path.suffix:
                new_filename = f"{base_path.stem}_{timestamp}{base_path.suffix}"
            else:
                new_filename = f"{base_path.name}_{timestamp}"

            output_path = base_path.parent / new_filename
        else:
            output_path = Path(self.config.output_collection_path)

        output_path.parent.mkdir(exist_ok=True)
        try:
            header = ""
            if self.config.enable_output_header:
                header = (
                    f"{'=' * 80}\nPROJECT COLLECTION\n"
                    f"Info: {result.info}\nTotal Files: {result.total_files}\n"
                    f"Time: {result.elapsed_time:.2f}s\n{'=' * 80}\n\n"
                )

            full_content = header + result.content
            output_path.write_text(full_content, encoding="utf-8")
            size_kb = output_path.stat().st_size / 1024
            print(f"\n✅ Collection saved to {output_path} ({size_kb:.1f} KB)")
        except IOError as e:
            print(f"❌ Error writing output file: {e}")


class ConsoleProgressListener:
    """Display collection progress in console."""

    def on_start(self, total: int):
        print(f"\n📁 Found {total} files to process...")

    def on_update(self, processed: int, total: int):
        print(f"\r🔄 Processing: {processed}/{total} files...", end="", flush=True)

    def on_complete(self, elapsed_time: float):
        print(f"\n✅ Collection complete in {elapsed_time:.2f}s.")


# ============================================================================
# USER INTERACTION
# ============================================================================

class UserSelectionParser:
    """Parse user input for folder and pattern selections."""

    def parse_folder_selection(self, sel_str: str, f_map: Dict) -> List[Dict[str, Any]]:
        selections = []
        for item in sel_str.split(","):
            item = item.strip()
            depth = CollectionDepth.DEEP if item.startswith("*") else CollectionDepth.SHALLOW
            try:
                folder_id = int(item.lstrip("*"))
                if folder_id in f_map:
                    selections.append({"path": f_map[folder_id]["path"], "depth": depth})
                else:
                    print(f"⚠️ Invalid folder ID: {folder_id}")
            except ValueError:
                print(f"⚠️ Invalid selection: '{item}'")
        return selections

    def parse_pattern_selection(self, sel_str: str, file_patterns: List[PatternDefinition]) -> List[str]:
        """Parse pattern selection from hybrid pattern definitions."""
        patterns = []
        pattern_strings = []
        
        input_parts = [part.strip() for part in sel_str.split(",") if part.strip()]
        
        for part in input_parts:
            # Check if it's a number (pattern index)
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(file_patterns):
                    selected_pattern = file_patterns[idx]
                    sub_patterns = [p.strip() for p in selected_pattern.pattern.split(",") if p.strip()]
                    patterns.append(f"[{idx}] {selected_pattern.name}")
                    pattern_strings.extend(sub_patterns)
                    print(f"  ✓ Selected [{idx}]: {selected_pattern.name}")
                else:
                    print(f"  ⚠️ Invalid pattern index: {idx}")
            
            # Check if it's a custom pattern
            elif any(c in part for c in ["*", "?", "[", "]", "{", "}", "/"]):
                patterns.append(f"Custom: {part}")
                pattern_strings.append(part)
                print(f"  ✓ Custom pattern: {part}")
            
            else:
                print(f"  ⚠️ Invalid pattern: '{part}'")
        
        # Remove duplicates while preserving order
        unique_pattern_strings = []
        seen = set()
        for pattern_str in pattern_strings:
            if pattern_str not in seen:
                seen.add(pattern_str)
                unique_pattern_strings.append(pattern_str)
        
        print(f"  📋 Final patterns to search: {unique_pattern_strings}")
        return unique_pattern_strings


# ============================================================================
# COMPONENT FACTORIES
# ============================================================================

class ComponentFactory:
    """Create application components."""

    @staticmethod
    def create_core_components(config: ProjectConfig) -> tuple:
        comment_remover = CommentRemover()
        pattern_loader = IgnorePatternLoader(config)
        file_discoverer = FileDiscoverer(config, pattern_loader)
        return comment_remover, pattern_loader, file_discoverer

    @staticmethod
    def create_ui_components(config: ProjectConfig) -> tuple:
        progress_listener = ConsoleProgressListener()
        output_writer = FileOutputWriter(config)
        selection_parser = UserSelectionParser()
        return progress_listener, output_writer, selection_parser


# ============================================================================
# APPLICATION ORCHESTRATOR
# ============================================================================

class ProjectManager:
    """Orchestrates the entire project management workflow."""
    
    def __init__(
        self,
        config_provider: ConfigProvider,
        tree_generator: TreeGenerator,
        file_collector: FileCollector,
        output_writer: OutputWriter,
        selection_parser: UserSelectionParser,
        initial_config: ProjectConfig,
    ):
        self.config_provider = config_provider
        self.tree_generator = tree_generator
        self.file_collector = file_collector
        self.output_writer = output_writer
        self.selection_parser = selection_parser
        self.config = initial_config

    def run(self):
        """Start main interactive loop."""
        self._show_header()
        while True:
            try:
                choice = self._show_enhanced_menu()

                if choice == "0":
                    print("Goodbye! 👋")
                    break
                elif choice == "1":
                    self.generate_full_structure()
                elif choice == "2":
                    self.generate_folders_only()
                elif choice == "3":
                    self.collect_by_folder()
                elif choice == "4":
                    self.collect_by_pattern()
                elif choice == "5":
                    self._settings_menu()
                elif choice == "?":
                    self._show_help()
                else:
                    print("❌ Invalid choice. Enter '?' for help.")

            except (KeyboardInterrupt, EOFError):
                print("\n\n⏹️Operation cancelled. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")

    def _show_header(self):
        print("\n" + "=" * 60)
        print("           🚀 ENHANCED PROJECT MANAGER")
        print("=" * 60)
        print(f"📂 Project Type: {self.config.project_type.upper()}")
        print(f"📁 Root: {Path.cwd().name}")
        print(f"📁 Base Folder: {self.config.base_project_folder}")
        print(f"🔒 Config Locked: {'✅ Yes' if self.config.lock_configuration else '❌ No'}")
        print("=" * 60)
        print("💡 Tip: Enter '?' for help at any time")
        print("=" * 60)

    def _show_enhanced_menu(self):
        """Display user-friendly menu."""
        print(f"\n🏠 MAIN MENU [{self.config.project_type.upper()}]")
        print("   [1] 📊 Generate Full Project Structure (Folders + Files)")
        print("   [2] 📁 Generate Folders Structure Only")
        print("   [3] 📂 Collect Files from Selected Folders")
        print("   [4] 🔍 Search Files by Type/Pattern")
        print("   [5] ⚙️  Settings & Configuration")
        print("   [0] 🚪 Exit")
        print("   [?] ❓ Help")

        return input("\n👉 Choose an option (0-5 or ?): ").strip()

    def _show_help(self):
        """Show context-sensitive help."""
        print("\n" + "💡" * 20)
        print("📖 QUICK HELP GUIDE")
        print("💡" * 20)
        print("""
[1] Full Structure: Shows complete project layout with ALL files
[2] Folders Only:   Shows only folder hierarchy (faster for large projects)
[3] Folder Collection: Select specific folders to scan and collect files
[4] Pattern Search:  Search for files by name patterns (e.g., *.py, test_*)
[5] Settings:       Configure tool behavior and project type

💡 Tips:
• Use numbers for quick folder selection
• Use * before number for deep scan (e.g., *3)
• Multiple selections: 1, *3, 5
• File patterns: *.py, *.html, test_*,features/**/widgets/*.dart
        """)

    def generate_full_structure(self):
        """Generate complete project structure with files."""
        print("🌳 Generating FULL project structure (folders + files)...")
        success = self.tree_generator.generate_file(Path.cwd(), include_files=True)
        if not success:
            print("❌ Failed to generate full structure.")

    def generate_folders_only(self):
        """Generate folders-only structure."""
        print("\n📁 Generating FOLDERS-ONLY structure...")
        success = self.tree_generator.generate_file(Path.cwd(), include_files=False)
        if not success:
            print("❌ Failed to generate folders structure.")

    def collect_by_folder(self):
        """Enhanced folder collection."""
        print("\n📂 FOLDER COLLECTION")
        folder_map = self.tree_generator.generate_interactive_map(Path.cwd())

        if len(folder_map) <= 1:
            print("ℹ️  No sub-folders found to select.")
            return

        print("\n💡 How to select:")
        print("   • Numbers only: Scan selected folder (SHALLOW)")
        print("   • * before number: Scan folder + subfolders (DEEP)")
        print("   • Examples: '1' or '*1' or '1, *3, 5'")

        selection_str = input("\n👉 Enter folder selection: ").strip()
        if not selection_str:
            print("⏹️  No selection made.")
            return

        if selection_str == "?":
            self._show_folder_help()
            return

        selections = self.selection_parser.parse_folder_selection(selection_str, folder_map)
        if not selections:
            print("❌ No valid folders selected.")
            return

        print(f"\n🚀 Starting collection for {len(selections)} folder(s)...")
        for sel in selections:
            depth_symbol = "📊 DEEP" if sel["depth"] == CollectionDepth.DEEP else "📁 SHALLOW"
            print(f"   - {sel['path'].name} ({depth_symbol})")

        start_time = time.monotonic()
        total_files, content = self.file_collector.collect_from_folders(selections)
        elapsed = time.monotonic() - start_time

        if total_files > 0:
            folder_names = [
                f"{s['path'].name}{'(*)' if s['depth'] == CollectionDepth.DEEP else ''}" for s in selections
            ]
            info = f"Folders: {', '.join(folder_names)}"
            result = CollectionResult(total_files, content, info, elapsed)
            self.output_writer.write(result)
            
        else:
            print("❌ No files found in selected folders.")

    def collect_by_pattern(self):
        """Enhanced pattern collection with hybrid pattern support."""
        print("\n🔍 FILE PATTERN COLLECTION")
        print("Available patterns (select by number or enter custom pattern):")
        
        if not self.config.file_patterns:
            print("ℹ️ No patterns configured for this project type.")
            available_patterns = []
        else:
            # Display patterns
            for i, pattern_def in enumerate(self.config.file_patterns):
                # Check if it's an individual pattern (name equals pattern)
                if pattern_def.name == pattern_def.pattern:
                    display_text = pattern_def.name
                    if len(display_text) > 50:
                        display_text = display_text[:47] + "..."
                    print(f"    [{i:2}] {display_text}")
                else:
                    # It's a grouped pattern
                    display_name = pattern_def.name
                    if len(display_name) > 60:
                        display_name = display_name[:57] + "..."
                    print(f"    [{i:2}] {display_name}")
            
            available_patterns = self.config.file_patterns
        
        print("\n💡 How to select:")
        print("    • Numbers: Select patterns by their number")
        print("    • Custom: Enter patterns like *.py, *.dart, *_model.dart")
        print("    • Multiple: Use commas to separate (e.g., 0, 5, 8)")
        
        selection_str = input("\n👉 Enter pattern selection: ").strip()
        if not selection_str:
            print("⏹️ No selection made.")
            return
        
        if selection_str == "?":
            self._show_pattern_help()
            return
        
        patterns = self.selection_parser.parse_pattern_selection(selection_str, available_patterns)
        if not patterns:
            print("❌ No valid patterns selected.")
            return
        
        print(f"\n🚀 Starting search for {len(patterns)} pattern(s)...")
        for i, pattern in enumerate(patterns):
            print(f"    [{i+1:2}] {pattern}")
        
        start_time = time.monotonic()
        total_files, content = self.file_collector.collect_by_patterns(patterns)
        elapsed = time.monotonic() - start_time
        
        if total_files > 0:
            result = CollectionResult(
                total_files, 
                content, 
                f"Patterns: {', '.join(patterns[:3])}" + ("..." if len(patterns) > 3 else ""), 
                elapsed
            )
            self.output_writer.write(result)
        else:
            print("❌ No files found matching the specified patterns.")

    def _show_folder_help(self):
        print("""
📖 FOLDER SELECTION HELP:
• Single number: Scan only the selected folder
   Example: '3' = Scan folder 3 only

• * before number: Deep scan (folder + all subfolders)
   Example: '*3' = Deep scan folder 3 and all contents

• Multiple selections: Combine different types
   Example: '1, *3, 5' = Mix of scan types
        """)

    def _show_pattern_help(self):
        print("""
📖 PATTERN SELECTION HELP:

Individual Patterns:
• Single file patterns (e.g., *_model.dart, *.py)
• Displayed as the pattern itself

Grouped Patterns:
• Multiple patterns with descriptive names
• Example: "Python: All Important Files"

Selection Methods:
1. By Number: '0, 2, 5' - Select patterns at positions 0, 2, and 5
2. Custom Pattern: '*.py, *_model.dart' - Enter exact patterns
3. Mixed: '0, *_screen.dart, 5' - Combine numbers and custom patterns

Examples:
• '0, 2, 8' - Select patterns 0, 2, and 8
• '*.dart, features/**/*.dart' - Custom recursive search
• '3, *_model.dart, 10' - Mixed selection
        """)

    def _settings_menu(self):
        """Enhanced settings menu."""
        while True:
            print(f"⚙️  SETTINGS MENU [{self.config.project_type}]")
            print("   [1] 📋 Show Current Settings")
            print("   [2] 🔄 Reset to Default Settings")
            print(f"   [3] {'🔓 Unlock' if self.config.lock_configuration else '🔒 Lock'} Configuration")
            print("   [4] 🎯 Change Project Type")
            print("   [5] 📁 Change Base Project Folder")
            print("   [0] ↩️  Back to Main Menu")

            choice = input("\n👉 Choose setting option: ").strip()

            if choice == "1":
                self._show_current_settings()
            elif choice == "2":
                self.config = self.config_provider.reset()
                print("✅ Settings reset to defaults!")
            elif choice == "3":
                self._toggle_config_lock()
            elif choice == "4":
                self._change_project_type()
            elif choice == "5":
                self._change_base_folder()
            elif choice == "0":
                break
            else:
                print("❌ Invalid choice.")

    def _show_current_settings(self):
        """Show settings in user-friendly format."""
        print("\n📋 CURRENT SETTINGS:")
        print(f"   Project Type: {self.config.project_type}")
        print(f"   Base Folder: {self.config.base_project_folder}")
        print(f"   Output File: {self.config.output_collection_path}")
        print(f"   Tree File: {self.config.output_tree_path}")
        print(f"   Config Lock: {'🔒 Locked' if self.config.lock_configuration else '🔓 Unlocked'}")
        print(f"   Max Threads: {self.config.max_worker_threads}")
        print(f"   Include TODOs: {not self.config.skip_todo_comments}")
        print(f"   Remove Comments: {self.config.remove_singleline_comment}")
        print(f"   File Extensions: {', '.join(self.config.allowed_extensions[:3])}...")
        print(f"   Ignore Patterns: {', '.join(self.config.ignore_patterns[:3])}...")

    def _toggle_config_lock(self):
        """Toggle configuration lock."""
        new_config_dict = self.config.to_dict()
        new_config_dict["lock_configuration"] = not self.config.lock_configuration
        self.config = ProjectConfig.from_dict(new_config_dict)
        self.config_provider.save(self.config)

        status = "🔒 Locked" if self.config.lock_configuration else "🔓 Unlocked"
        print(f"✅ Configuration is now {status}.")

    def _change_project_type(self):
        """Change project type manually."""
        print("\n🎯 AVAILABLE PROJECT TYPES:")
        for i, ptype in enumerate(ProjectType):
            print(f"   [{i}] {ptype.value}")

        try:
            choice = input("\n👉 Select project type (number): ").strip()
            if not choice:
                return

            choice_idx = int(choice)
            if 0 <= choice_idx < len(ProjectType):
                selected_type = list(ProjectType)[choice_idx]
                new_config = ProjectConfigFactory.create(selected_type)
                self.config_provider.save(new_config)
                self.config = new_config
                print(f"✅ Project type changed to {selected_type.value}!")
            else:
                print("❌ Invalid project type number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    def _change_base_folder(self):
        """Change base project folder."""
        print(f"\n📁 CURRENT BASE FOLDER: {self.config.base_project_folder}")
        print("💡 Enter new base folder path (relative to project root):")
        print("   Examples: '.', 'lib', 'src', 'app'")

        new_folder = input("👉 New base folder: ").strip()
        if not new_folder:
            print("⏹️ No changes made.")
            return

        test_path = Path.cwd() / new_folder
        if not test_path.exists() or not test_path.is_dir():
            print(f"❌ Folder '{new_folder}' does not exist or is not a directory!")
            return

        new_config_dict = self.config.to_dict()
        new_config_dict["base_project_folder"] = new_folder
        self.config = ProjectConfig.from_dict(new_config_dict)
        self.config_provider.save(self.config)

        print(f"✅ Base project folder changed to '{new_folder}'!")


# ============================================================================
# COMPOSITION ROOT & ENTRY POINT
# ============================================================================

class AppFactory:
    """Factory to compose and create main application instance."""

    @staticmethod
    def create() -> ProjectManager:
        """Detect project type and wire up all dependencies."""
        root = Path.cwd()

        settings_path = root / "output" / "settings.json"
        config_provider = JsonConfigProvider(settings_path)
        final_config = ConfigManager.create_final_config(config_provider)

        comment_formatter = CommentFormatter()
        content_organizer = ContentOrganizer(comment_formatter)

        comment_remover, pattern_loader, file_discoverer = ComponentFactory.create_core_components(final_config)
        progress_listener, output_writer, selection_parser = ComponentFactory.create_ui_components(final_config)

        file_processor = FileProcessor(final_config, comment_remover)
        tree_generator = EnhancedTreeGenerator(final_config, file_discoverer)
        file_collector = ParallelFileCollector(file_processor, file_discoverer, progress_listener, content_organizer)

        return ProjectManager(
            config_provider=config_provider,
            tree_generator=tree_generator,
            file_collector=file_collector,
            output_writer=output_writer,
            selection_parser=selection_parser,
            initial_config=final_config,
        )


def main():
    """Main entry point for the application."""
    try:
        manager = AppFactory.create()
        manager.run()
    except Exception as e:
        print(f"\n❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
        if project_root.is_dir():
            os.chdir(project_root)
        else:
            print(f"❌ Error: Provided path '{project_root}' is not a valid directory.")
            sys.exit(1)

    main()
