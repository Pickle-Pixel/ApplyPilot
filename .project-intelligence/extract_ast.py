import ast
import json
import os
import re
import hashlib
from pathlib import Path

REPO_ROOT = Path(r"d:\obsidian\4_项目\2025逃跑计划\ApplyPilot")
SRC_ROOT = REPO_ROOT / "src" / "applypilot"
OUT_DIR = REPO_ROOT / ".project-intelligence" / "inventory"

CATEGORY_RULES = [
    ("config", lambda p: "config" in p.parts),
    ("discovery", lambda p: "discovery" in p.parts),
    ("enrichment", lambda p: "enrichment" in p.parts),
    ("scoring", lambda p: "scoring" in p.parts),
    ("apply", lambda p: "apply" in p.parts),
    ("wizard", lambda p: "wizard" in p.parts),
    ("core", lambda p: True),
]


def assign_category(rel_path: Path) -> str:
    for cat, rule in CATEGORY_RULES:
        if rule(rel_path):
            return cat
    return "other"


def get_source_type(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    return "source"


def extract_expr(node) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def parse_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict]:
    params = []
    all_args = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(all_args) - len(node.args.defaults)) + list(node.args.defaults)
    for i, arg in enumerate(all_args):
        ann = extract_expr(arg.annotation)
        default = extract_expr(defaults[i]) if i < len(defaults) else None
        params.append({
            "name": arg.arg,
            "type_annotation": ann,
            "default": default,
        })
    if node.args.vararg:
        params.append({
            "name": f"*{node.args.vararg.arg}",
            "type_annotation": extract_expr(node.args.vararg.annotation),
            "default": None,
        })
    kw_defaults = [None] * (len(node.args.kwonlyargs) - len(node.args.kw_defaults)) + list(node.args.kw_defaults)
    for i, arg in enumerate(node.args.kwonlyargs):
        params.append({
            "name": arg.arg,
            "type_annotation": extract_expr(arg.annotation),
            "default": extract_expr(kw_defaults[i]) if i < len(kw_defaults) else None,
        })
    if node.args.kwarg:
        params.append({
            "name": f"**{node.args.kwarg.arg}",
            "type_annotation": extract_expr(node.args.kwarg.annotation),
            "default": None,
        })
    return params


def collect_decorators(node) -> list[str]:
    out = []
    for d in node.decorator_list:
        txt = extract_expr(d)
        if txt:
            out.append(txt)
    return out


def walk_module(tree: ast.Module, rel_path: str) -> list[dict]:
    symbols = []
    mod_id = f"FILE-{hashlib.md5(rel_path.encode()).hexdigest()[:8]}"
    counter = [0]

    def next_sym_id(kind: str) -> str:
        counter[0] += 1
        return f"SYM-{hashlib.md5((rel_path + '-' + kind + '-' + str(counter[0])).encode()).hexdigest()[:8]}"

    def visit(node, parent_qname=None):
        if isinstance(node, ast.ClassDef):
            sym_id = next_sym_id("class")
            qname = f"{parent_qname}.{node.name}" if parent_qname else node.name
            symbols.append({
                "symbol_id": sym_id,
                "kind": "class",
                "name": node.name,
                "file_id": mod_id,
                "qualified_name": qname,
                "parent_symbol": parent_qname,
                "parameters": [],
                "return_type": None,
                "docstring": ast.get_docstring(node),
                "decorators": collect_decorators(node),
                "line_start": getattr(node, "lineno", None),
                "line_end": getattr(node, "end_lineno", None),
            })
            for child in node.body:
                visit(child, qname)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            sym_id = next_sym_id(kind)
            qname = f"{parent_qname}.{node.name}" if parent_qname else node.name
            symbols.append({
                "symbol_id": sym_id,
                "kind": kind,
                "name": node.name,
                "file_id": mod_id,
                "qualified_name": qname,
                "parent_symbol": parent_qname,
                "parameters": parse_signature(node),
                "return_type": extract_expr(node.returns),
                "docstring": ast.get_docstring(node),
                "decorators": collect_decorators(node),
                "line_start": getattr(node, "lineno", None),
                "line_end": getattr(node, "end_lineno", None),
            })
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, parent_qname)

    for node in tree.body:
        visit(node, None)

    mod_sym_id = f"SYM-{hashlib.md5((rel_path + '-module').encode()).hexdigest()[:8]}"
    symbols.insert(0, {
        "symbol_id": mod_sym_id,
        "kind": "module",
        "name": Path(rel_path).stem,
        "file_id": mod_id,
        "qualified_name": Path(rel_path).stem,
        "parent_symbol": None,
        "parameters": [],
        "return_type": None,
        "docstring": ast.get_docstring(tree),
        "decorators": [],
        "line_start": 1,
        "line_end": len(ast.unparse(tree).splitlines()),
    })
    return symbols, mod_id


def extract_imports(tree: ast.Module) -> list[dict]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "as_name": alias.asname,
                    "line": node.lineno,
                    "type": "import",
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                    "from_module": node.module,
                    "as_name": alias.asname,
                    "line": node.lineno,
                    "type": "from_import",
                })
    return imports


def scan_python_files():
    files = []
    symbols_all = []
    imports_all = []

    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        rel = py_file.relative_to(REPO_ROOT)
        rel_str = str(rel).replace("\\", "/")
        category = assign_category(rel)
        src_type = get_source_type(py_file)
        file_id = f"FILE-{hashlib.md5(rel_str.encode()).hexdigest()[:8]}"

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            syms, fid = walk_module(tree, rel_str)
            imps = extract_imports(tree)
            files.append({
                "file_id": fid,
                "path": rel_str,
                "language": "python",
                "category": category,
                "source_type": src_type,
                "parse_status": "parsed",
                "symbol_ids": [s["symbol_id"] for s in syms],
                "line_count": source.count("\n") + 1,
                "evidence": {
                    "method": "python_ast",
                },
            })
            symbols_all.extend(syms)
            imports_all.append({
                "file_id": fid,
                "path": rel_str,
                "imports": imps,
            })
        except SyntaxError as e:
            files.append({
                "file_id": file_id,
                "path": rel_str,
                "language": "python",
                "category": category,
                "source_type": src_type,
                "parse_status": "failed",
                "symbol_ids": [],
                "error": str(e),
            })
        except Exception as e:
            files.append({
                "file_id": file_id,
                "path": rel_str,
                "language": "python",
                "category": category,
                "source_type": src_type,
                "parse_status": "error",
                "symbol_ids": [],
                "error": str(e),
            })

    return files, symbols_all, imports_all


def scan_root_configs():
    extras = [
        ("pyproject.toml", "config"),
        (".gitignore", "config"),
        (".env.example", "config"),
        ("profile.example.json", "config"),
        ("README.md", "doc"),
        ("CHANGELOG.md", "doc"),
        ("CONTRIBUTING.md", "doc"),
        ("LICENSE", "doc"),
        ("ApplyPilot_Reverse_Product_Discovery_Playbook.md", "doc"),
    ]
    root_files = []
    for name, cat in extras:
        p = REPO_ROOT / name
        if p.exists():
            rel_str = name
            file_id = f"FILE-{hashlib.md5(rel_str.encode()).hexdigest()[:8]}"
            root_files.append({
                "file_id": file_id,
                "path": rel_str,
                "language": "markdown" if name.endswith(".md") or name == "LICENSE" else (
                    "toml" if name.endswith(".toml") else (
                    "json" if name.endswith(".json") else (
                    "yaml" if name.endswith(".yaml") or name.endswith(".yml") else "text"))),
                "category": cat,
                "source_type": "config" if cat == "config" else "doc",
                "parse_status": "recorded",
                "symbol_ids": [],
            })

    for workflow in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        rel = workflow.relative_to(REPO_ROOT)
        rel_str = str(rel).replace("\\", "/")
        file_id = f"FILE-{hashlib.md5(rel_str.encode()).hexdigest()[:8]}"
        root_files.append({
            "file_id": file_id,
            "path": rel_str,
            "language": "yaml",
            "category": "ci",
            "source_type": "config",
            "parse_status": "recorded",
            "symbol_ids": [],
        })

    yaml_configs = []
    conf_dir = SRC_ROOT / "config"
    for yf in conf_dir.glob("*.yaml"):
        rel = yf.relative_to(REPO_ROOT)
        rel_str = str(rel).replace("\\", "/")
        file_id = f"FILE-{hashlib.md5(rel_str.encode()).hexdigest()[:8]}"
        try:
            import yaml as pyyaml
            data = pyyaml.safe_load(yf.read_text(encoding="utf-8"))
            yaml_configs.append({
                "file_id": file_id,
                "path": rel_str,
                "structure_keys": list(data.keys()) if isinstance(data, dict) else (
                    f"list[{len(data)}]" if isinstance(data, list) else str(type(data).__name__)),
            })
        except ImportError:
            yaml_configs.append({
                "file_id": file_id,
                "path": rel_str,
                "structure_keys": "pyyaml_not_available",
            })
        except Exception as e:
            yaml_configs.append({
                "file_id": file_id,
                "path": rel_str,
                "structure_keys": f"parse_error: {e}",
            })

    return root_files, yaml_configs


def main():
    files, symbols, imports = scan_python_files()
    root_files, yaml_configs = scan_root_configs()
    all_files = files + root_files

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files_out = {"schema_version": "1.0", "generated_at": "2026-11-07", "files": all_files}
    (OUT_DIR / "files.json").write_text(json.dumps(files_out, indent=2, ensure_ascii=False), encoding="utf-8")

    symbols_out = {"schema_version": "1.0", "generated_at": "2026-11-07", "symbols": symbols}
    (OUT_DIR / "symbols.json").write_text(json.dumps(symbols_out, indent=2, ensure_ascii=False), encoding="utf-8")

    imports_out = {"schema_version": "1.0", "generated_at": "2026-11-07", "imports": imports}
    (OUT_DIR / "imports.json").write_text(json.dumps(imports_out, indent=2, ensure_ascii=False), encoding="utf-8")

    configs_out = {"schema_version": "1.0", "generated_at": "2026-11-07", "yaml_configs": yaml_configs}
    (OUT_DIR / "configs.json").write_text(json.dumps(configs_out, indent=2, ensure_ascii=False), encoding="utf-8")

    parsed = sum(1 for f in files if f["parse_status"] == "parsed")
    failed = sum(1 for f in files if f["parse_status"] != "parsed")
    sym_counts = {}
    for s in symbols:
        sym_counts[s["kind"]] = sym_counts.get(s["kind"], 0) + 1

    report = {
        "python_files_total": len(files),
        "python_parsed": parsed,
        "python_failed": failed,
        "root_configs": len(root_files),
        "yaml_configs": len(yaml_configs),
        "symbol_counts": sym_counts,
        "files_by_category": {},
    }
    for f in all_files:
        c = f["category"]
        report["files_by_category"][c] = report["files_by_category"].get(c, 0) + 1

    (OUT_DIR.parent / "reports" / "coverage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()