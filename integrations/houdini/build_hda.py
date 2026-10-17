"""Run with Houdini's hython, or exec this file in Houdini's Python shell."""

import sys
from pathlib import Path

import hou


def build(destination=None):
    here = Path(__file__).resolve().parent
    output = Path(destination) if destination else here / "image2point_proxy.hda"
    if hou.licenseCategory() == hou.licenseCategoryType.Apprentice:
        output = output.with_suffix(".hdanc")
    if output.exists():
        raise FileExistsError("Preserving existing asset: " + str(output))
    # Release ZIPs copy point_package.py beside this file; a repository checkout keeps it in src/.
    package = here / "point_package.py"
    if not package.exists():
        package = here.parents[1] / "src/image2point/point_package.py"
    source = package.read_text()
    module = source + "\n" + (here / "hda_runtime.py").read_text()
    container = hou.node("/obj").createNode("geo", "image2point_builder", run_init_scripts=False)
    try:
        subnet = container.createNode("subnet", "proxy")
        # Output 0 (Render) is what a Geometry ROP and Local Bake write: full resolution, no
        # viewport checks. Output 1 (Preview) adds Preview Point Spacing and the checks. Each has
        # its own Python SOP, so the viewport only cooks the light Preview.
        for index, (name, bake) in enumerate((("RENDER", True), ("PREVIEW", False))):
            python = subnet.createNode("python", name)
            python.parm("python").set(
                "node = hou.pwd()\n"
                "try:\n"
                f"    node.parent().hdaModule().geometry(node.parent(), node.geometry(), bake={bake})\n"
                "except Exception as error:\n"
                "    raise hou.NodeError(str(error))\n"
            )
            output_node = subnet.createNode("output", "OUT_" + name)
            output_node.parm("outputidx").set(index)
            output_node.setInput(0, python)
            if index == 0:
                output_node.setDisplayFlag(True)
                output_node.setRenderFlag(True)
        node = subnet.createDigitalAsset(
            name="image2point::proxy::1.0",
            hda_file_name=str(output),
            description="Image2Point Proxy",
            min_num_inputs=0,
            max_num_inputs=0,
        )
        node.type().definition().setMaxNumOutputs(2)
        group = hou.ParmTemplateGroup()
        manifest = hou.StringParmTemplate(
            "manifest",
            "Point Cloud Cache",
            1,
            string_type=hou.stringParmType.FileReference,
            help="Sets the bake range only. Scene FPS and timeline remain controlled by your shot setup.",
        )
        manifest.setScriptCallback("kwargs['node'].hdaModule().manifest_changed(kwargs['node'])")
        manifest.setScriptCallbackLanguage(hou.scriptLanguage.Python)
        group.append(manifest)
        group.append(
            hou.FloatParmTemplate(
                "offset",
                "Depth Offset (metres)",
                1,
                default_value=(0.0,),
                # Slider range only; typed values beyond it are allowed.
                min=-1.0,
                max=1.0,
                min_is_strict=False,
                max_is_strict=False,
                help="Moves along each camera ray, preserving plate alignment. Negative is closer to camera.",
            )
        )
        group.append(
            hou.ToggleParmTemplate(
                "stabilize",
                "Stabilize Depth",
                default_value=True,
                help=(
                    "Removes the frame-to-frame depth 'breathing' of MoGe by matching each frame's scale to "
                    "its neighbours, measured on the static background. Moves points along camera rays, so "
                    "plate alignment is kept. The first cook reads every frame once."
                ),
            )
        )
        group.append(
            hou.IntParmTemplate(
                "bg_smoothing",
                "Background Smoothing (frames)",
                1,
                default_value=(0,),
                min=0,
                max=8,
                help=(
                    "Averages each background point with the same spot in this many frames on each side, "
                    "matched through the tracked camera, to calm surface shimmer on the static set. 0 is off. "
                    "Each step reads two more frames per cook, so cooks and bakes get slower."
                ),
            )
        )
        group.append(hou.IntParmTemplate("inset", "Edge Repair (grid pixels)", 1, default_value=(1,), min=0, max=64))
        group.append(
            hou.IntParmTemplate("extension", "Edge Extend (grid pixels)", 1, default_value=(1,), min=0, max=64)
        )
        group.append(
            hou.FloatParmTemplate(
                "tolerance",
                "Depth Tolerance (metres)",
                1,
                default_value=(0.0,),
                min=0.0,
                max=5.0,
                min_is_strict=True,
                max_is_strict=False,
                help=(
                    "Actor pixels further than this from the actor's median depth are treated as unreliable "
                    "(blended edge depth) and take the nearest reliable depth. 0 (default) turns it off. Don't "
                    "use it on actors that span a long depth: it flattens them."
                ),
            )
        )
        group.append(hou.FloatParmTemplate("coverage", "Point Coverage", 1, default_value=(1.05,), min=0.01))
        group.append(
            hou.IntParmTemplate(
                "preview_stride",
                "Preview Point Spacing",
                1,
                default_value=(4,),
                min=1,
                max=16,
                help="Preview only. Bake always uses all available cache samples.",
            )
        )
        group.append(hou.ToggleParmTemplate("remove_background", "Remove Background", default_value=False))
        per_actor = hou.FolderParmTemplate(
            "actor_controls", "Per-Actor Overrides", folder_type=hou.folderType.MultiparmBlock
        )
        # Adding a row gives it the next unused actor ID instead of repeating ID 1.
        per_actor.setScriptCallback("kwargs['node'].hdaModule().rows_changed(kwargs['node'])")
        per_actor.setScriptCallbackLanguage(hou.scriptLanguage.Python)
        per_actor.addParmTemplate(hou.IntParmTemplate("actor_id#", "Actor ID", 1, default_value=(1,), min=1))
        per_actor.addParmTemplate(
            hou.ToggleParmTemplate(
                "keep#",
                "Keep",
                default_value=True,
                help="Untick to leave this actor out. Once any row exists, actors without a row are left out too.",
            )
        )
        for name in ("offset", "inset", "extension", "tolerance", "coverage"):
            template = group.find(name)
            template.setName(name + "#")
            per_actor.addParmTemplate(template)
        per_actor.addParmTemplate(
            hou.IntParmTemplate(
                "smoothing#",
                "Depth Smoothing (frames)",
                1,
                default_value=(0,),
                min=0,
                max=12,
                help=(
                    "Smooths this actor's own frame-to-frame depth wobble over about this many frames. "
                    "0 is off. Keep it low for fast movement toward or away from camera, which it softens."
                ),
            )
        )
        mask = hou.FolderParmTemplate("mask_section", "Viewport Only", folder_type=hou.folderType.Simple)
        mask.addParmTemplate(
            hou.ToggleParmTemplate(
                "roto_check",
                "Roto Check",
                default_value=False,
                disable_when="{ id_check == 1 }",
                help="Viewport only: roto as point colour, not opacity. Black points remain visible. Not baked.",
            )
        )
        mask.addParmTemplate(
            hou.ToggleParmTemplate(
                "id_check",
                "ID Check",
                default_value=False,
                disable_when="{ roto_check == 1 }",
                help="Viewport only: one colour per actor, background grey. Not baked.",
            )
        )
        actors = hou.FolderParmTemplate("actors_section", "Actors", folder_type=hou.folderType.Tabs)
        actors.addParmTemplate(per_actor)
        bake = hou.FolderParmTemplate(
            "bake_section", "Local Bake", folder_type=hou.folderType.Tabs, ends_tab_group=True
        )
        bake.addParmTemplate(
            hou.StringParmTemplate(
                "output_file",
                "Output File",
                1,
                default_value=("$HIP/image2point/points.$F4.bgeo.sc",),
                string_type=hou.stringParmType.FileReference,
                help=(
                    "One file per frame; include a frame number such as $F4. Existing files are overwritten. "
                    "For the farm, use a Geometry ROP on the Render output (output 0) instead."
                ),
            )
        )
        for name, label, value in [("start", "Start Frame", 1001), ("end", "End Frame", 1024)]:
            bake.addParmTemplate(hou.IntParmTemplate(name, label, 1, default_value=(value,)))
        button = hou.ButtonParmTemplate("bake", "Bake Locally")
        button.setScriptCallback("kwargs['node'].hdaModule().bake_locally(kwargs['node'])")
        button.setScriptCallbackLanguage(hou.scriptLanguage.Python)
        bake.addParmTemplate(button)
        edges = hou.FolderParmTemplate(
            "edge_section",
            "Edge Treatment",
            (group.find("inset"), group.find("extension")),
            folder_type=hou.folderType.Simple,
        )
        adjust = hou.FolderParmTemplate(
            "adjust_section",
            "Global",
            (
                group.find("offset"),
                hou.FolderParmTemplate(
                    "stabilisation_section",
                    "Stabilisation",
                    (group.find("stabilize"), group.find("bg_smoothing")),
                    folder_type=hou.folderType.Simple,
                ),
                edges,
                group.find("coverage"),
                group.find("remove_background"),
            ),
            folder_type=hou.folderType.Tabs,
        )
        display = hou.FolderParmTemplate(
            "display_section",
            "Display",
            (mask, group.find("preview_stride")),
            folder_type=hou.folderType.Tabs,
        )
        group = hou.ParmTemplateGroup((manifest, adjust, actors, display, bake))
        definition = node.type().definition()
        definition.setParmTemplateGroup(group)
        definition.addSection("PythonModule", module)
        definition.addSection(
            "Help",
            "Loads Image2Point full-frame point packages. Use Y-up metre scene space. "
            "Output 0 (Render) is full resolution for Geometry ROPs and Local Bake; output 1 (Preview) "
            "applies Preview Point Spacing, Roto Check and ID Check for the viewport. "
            "Depth controls are camera-ray adjustments. "
            "roto (when the package has one), pscale and object_id are baked point attributes. "
            "RenderMan material binding must be validated in your studio. See USAGE.md.",
        )
        definition.updateFromNode(node)
        definition.save(str(output))
        print("Created " + str(output))
        return output
    finally:
        container.destroy()


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
