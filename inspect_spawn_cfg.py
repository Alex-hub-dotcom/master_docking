# inspect_camera_api.py
import sys, inspect, importlib, textwrap
print("="*80)
print("🔎 Isaac Lab Camera API – inspeção precisa")
print("="*80)

def safe_import(mod):
    try:
        return importlib.import_module(mod), None
    except Exception as e:
        return None, e

mods = [
    "isaaclab.sensors.camera",
    "isaacsim.sensors.camera",         # às vezes existe nessa build
    "omni.isaac.sensor",               # sensores legacy
]

for m in mods:
    mod, err = safe_import(m)
    print("\n" + "-"*80)
    print(f"📦 módulo: {m}")
    if err:
        print(f"  ❌ não importou: {err}")
        continue
    print(f"  ✅ importado: {mod!r}")
    names = dir(mod)
    # tenta achar classes Camera/CameraCfg
    cam_cls = getattr(mod, "Camera", None)
    cfg_cls = getattr(mod, "CameraCfg", None)

    if cam_cls:
        print("\n  ▶️ Classe Camera encontrada:", cam_cls)
        try:
            sig = inspect.signature(cam_cls.__init__)
            print("  ├─ assinatura __init__:", sig)
        except Exception as e:
            print("  ├─ não consegui ler assinatura:", e)

        # Procura métodos úteis (from_prim, attach, create, etc)
        candidates = [n for n in names if "camera" in n.lower() or "prim" in n.lower()]
        extras = []
        for n in candidates:
            obj = getattr(mod, n)
            if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isclass(obj):
                extras.append(n)
        if extras:
            print("  ├─ utilitários relacionados:", ", ".join(sorted(set(extras))[:30]))
        # Procura métodos estáticos/comuns na classe
        cls_members = [n for n,_ in inspect.getmembers(cam_cls)]
        suggest = [n for n in cls_members if any(k in n.lower() for k in ["from", "create", "build", "attach", "spawn", "prim"])]
        if suggest:
            print("  ├─ membros interessantes na classe Camera:", ", ".join(sorted(set(suggest))[:30]))
    else:
        print("  ⚠️ Classe Camera **não** encontrada nesse módulo.")

    if cfg_cls:
        print("\n  ▶️ Classe CameraCfg encontrada:", cfg_cls)
        try:
            print("  ├─ é dataclass?:", inspect.isdataclass(cfg_cls))
            fields = getattr(cfg_cls, "__dataclass_fields__", {}) or {}
            if fields:
                print("  ├─ campos do CameraCfg:")
                for k, f in fields.items():
                    ftype = getattr(f.type, "__name__", str(f.type))
                    has_default = f.default is not inspect._empty
                    has_def_fact = f.default_factory is not inspect._empty  # type: ignore
                    default_repr = None
                    if has_default:
                        default_repr = repr(f.default)
                    elif has_def_fact:
                        default_repr = f"default_factory={f.default_factory!r}"
                    else:
                        default_repr = "<sem default>"
                    print(f"      - {k}: type={ftype} | {default_repr}")
            else:
                print("  ├─ não expõe __dataclass_fields__ (talvez não seja dataclass).")
        except Exception as e:
            print("  ├─ falha ao inspecionar CameraCfg:", e)
    else:
        print("  ⚠️ CameraCfg **não** encontrada nesse módulo.")

print("\n" + "="*80)
print("🧪 Teste rápido: existe algum '*SpawnCfg' em todo o pacote isaaclab?")
print("="*80)
import pkgutil
spawn_like = []
try:
    import isaaclab
    for m in pkgutil.walk_packages(isaaclab.__path__, isaaclab.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
            for n, obj in inspect.getmembers(mod, inspect.isclass):
                if n.endswith("SpawnCfg"):
                    spawn_like.append(f"{m.name}.{n}")
        except Exception:
            pass
except Exception as e:
    print("Não consegui varrer pacote isaaclab:", e)

if spawn_like:
    print("  ✅ Achei classes SpawnCfg:")
    for s in sorted(set(spawn_like)):
        print("   •", s)
else:
    print("  ⚠️ Nenhuma classe '*SpawnCfg' encontrada nessa instalação.")

print("\n" + "="*80)
print("✅ Fim. Rode:  python inspect_camera_api.py  e cola aqui o output.")
print("="*80)
