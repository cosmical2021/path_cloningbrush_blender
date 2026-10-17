bl_info = {
    "name": "Ping-Pong Path Texture Cloner (Loop & Eraser)",
    "author": "Gemini",
    "version": (2, 3, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > N-Panel > Path Clone | Image Editor > N-Panel > Path Clone",
    "description": "Reaaliaikainen tekstuurin polkukloonaus ping-pong- tai loop-tyyliin, Rake-kierrolla ja polun poistolla.",
    "category": "Paint",
}

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.interpolate import poly_3d_calc
from bpy_extras import view3d_utils
import numpy as np
import math
import time

# =====================================================================
# YHTEINEN DATA & APUFUNKTIOT
# =====================================================================

class ClonePathStore:
    uv_path = []           # Lähdepolku UV-koordinaateissa [Vector(u, v), ...]
    pts_3d = []            # Lähdepolku 3D-maailmankoordinaateissa
    spacing_uv = 0.002     # Näytteistysväli UV-avaruudessa
    current_uv = None      # Aktiivinen lukupiste UV-avaruudessa
    current_3d = None      # Aktiivinen lukupiste 3D-avaruudessa
    brush_3d = None        # Siveltimen nykyinen sijainti 3D-pinnalla
    brush_dir_3d = None    # Siveltimen nykyinen liikesuunta 3D-pinnalla
    draw_handle_2d = None
    draw_handle_3d = None

store = ClonePathStore()


def get_shader():
    try:
        return gpu.shader.from_builtin('UNIFORM_COLOR')
    except ValueError:
        return gpu.shader.from_builtin('2D_UNIFORM_COLOR')


def resample_path_with_3d(uv_pts, pts_3d, spacing, loop=False):
    """Tasavälistää UV-polun (ja 3D-polun). Jos loop=True, sulkee silmukan."""
    if len(uv_pts) < 2:
        return list(uv_pts), list(pts_3d)

    # Jos kyseessä on loop, lisätään alkupiste loppuun, jotta saadaan suljettu kehä
    working_uv = list(uv_pts)
    working_3d = list(pts_3d)
    has_3d = (len(pts_3d) == len(uv_pts))

    if loop:
        if (working_uv[0] - working_uv[-1]).length > 1e-5:
            working_uv.append(working_uv[0].copy())
            if has_3d:
                working_3d.append(working_3d[0].copy())

    res_uv = [working_uv[0].copy()]
    res_3d = [working_3d[0].copy()] if has_3d else []

    for i in range(1, len(working_uv)):
        p0_uv = res_uv[-1]
        p1_uv = working_uv[i]
        p0_3d = res_3d[-1] if has_3d else None
        p1_3d = working_3d[i] if has_3d else None

        dist = (p1_uv - p0_uv).length
        while dist >= spacing:
            t = spacing / dist
            p0_uv = p0_uv.lerp(p1_uv, t)
            res_uv.append(p0_uv)
            if has_3d:
                p0_3d = p0_3d.lerp(p1_3d, t)
                res_3d.append(p0_3d)
            dist = (p1_uv - p0_uv).length

    return res_uv, res_3d


def sample_path_mode(uv_path, pts_3d, distance, spacing, loop=False):
    """Palauttaa pisteen (UV, tangentti, 3D) joko Ping-Pong- tai Loop-tilassa."""
    if not uv_path:
        return Vector((0, 0)), Vector((1, 0)), None
    if len(uv_path) == 1:
        p3d = pts_3d[0] if pts_3d else None
        return uv_path[0], Vector((1, 0)), p3d

    has_3d = (len(pts_3d) == len(uv_path))

    if loop:
        # Loop-tila: kierretään rataa jatkuvasti samaan suuntaan ympäri
        total_len = len(uv_path) * spacing
        if total_len <= 1e-8:
            p3d = pts_3d[0] if pts_3d else None
            return uv_path[0], Vector((1, 0)), p3d

        d = distance % total_len
        idx_float = d / spacing
        idx = int(idx_float)
        frac = idx_float - idx

        next_idx = (idx + 1) % len(uv_path)
        uv = uv_path[idx].lerp(uv_path[next_idx], frac)
        tan = (uv_path[next_idx] - uv_path[idx]).normalized()
        p3d = pts_3d[idx].lerp(pts_3d[next_idx], frac) if has_3d else None

        if tan.length_squared == 0:
            tan = Vector((1, 0))
        return uv, tan, p3d

    else:
        # Ping-Pong-tila: edestakaisin alku- ja loppupisteen välillä
        total_len = (len(uv_path) - 1) * spacing
        if total_len <= 1e-8:
            p3d = pts_3d[0] if pts_3d else None
            return uv_path[0], Vector((1, 0)), p3d

        cycle = 2.0 * total_len
        m = distance % cycle
        forward = True
        if m > total_len:
            d = cycle - m
            forward = False
        else:
            d = m

        idx_float = d / spacing
        idx = int(idx_float)
        frac = idx_float - idx

        if idx >= len(uv_path) - 1:
            uv = uv_path[-1]
            tan = (uv_path[-1] - uv_path[-2]).normalized()
            p3d = pts_3d[-1] if has_3d else None
        else:
            uv = uv_path[idx].lerp(uv_path[idx + 1], frac)
            tan = (uv_path[idx + 1] - uv_path[idx]).normalized()
            p3d = pts_3d[idx].lerp(pts_3d[idx + 1], frac) if has_3d else None

        if not forward:
            tan = -tan
        if tan.length_squared == 0:
            tan = Vector((1, 0))

        return uv, tan, p3d


def find_target_image(context):
    props = context.scene.pp_tex_clone_props
    if props.target_image and props.target_image.has_data:
        return props.target_image

    obj = context.active_object
    if obj and obj.type == 'MESH' and obj.active_material:
        mat = obj.active_material
        if mat.use_nodes and mat.node_tree:
            nodes = mat.node_tree.nodes
            if nodes.active and nodes.active.type == 'TEX_IMAGE' and nodes.active.image:
                return nodes.active.image
            for n in nodes:
                if n.type == 'TEX_IMAGE' and n.image and n.image.has_data:
                    return n.image

    if context.area and context.area.type == 'IMAGE_EDITOR':
        img = context.space_data.image
        if img and img.has_data:
            return img

    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                img = area.spaces.active.image
                if img and img.has_data:
                    return img

    return None


def raycast_uv_on_active_mesh(context, event):
    obj = context.active_object
    if not obj or obj.type != 'MESH':
        return False, None, None

    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return False, None, None

    coord = (event.mouse_region_x, event.mouse_region_y)
    view_vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

    mat_inv = obj.matrix_world.inverted()
    ray_origin_obj = mat_inv @ ray_origin
    ray_target_obj = mat_inv @ (ray_origin + view_vec)
    ray_dir_obj = (ray_target_obj - ray_origin_obj).normalized()

    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.data

    if not mesh.uv_layers.active:
        return False, None, None

    success, loc_obj, norm_obj, face_idx = obj_eval.ray_cast(ray_origin_obj, ray_dir_obj)
    if not success or face_idx < 0:
        return False, None, None

    poly = mesh.polygons[face_idx]
    uv_layer = mesh.uv_layers.active.data

    v_coords = [mesh.vertices[mesh.loops[loop_idx].vertex_index].co for loop_idx in poly.loop_indices]
    uv_coords = [uv_layer[loop_idx].uv for loop_idx in poly.loop_indices]

    weights = poly_3d_calc(v_coords, loc_obj)
    uv_hit = Vector((0.0, 0.0))
    for w, uv in zip(weights, uv_coords):
        uv_hit += w * Vector((uv.x, uv.y))

    world_pos = obj.matrix_world @ (loc_obj + norm_obj * 0.002)
    return True, world_pos, uv_hit


# =====================================================================
# NUMPY-PIKSELIKLOONAAJA & RAKE
# =====================================================================

class FastPixelCloner:
    def __init__(self, image):
        self.image = image
        self.width, self.height = image.size
        raw = np.empty(self.width * self.height * 4, dtype=np.float32)
        image.pixels.foreach_get(raw)
        self.src_buf = raw.reshape((self.height, self.width, 4))
        self.dst_buf = self.src_buf.copy()
        self.is_dirty = False
        self.last_push_time = 0.0

    def commit_stroke_to_source(self):
        np.copyto(self.src_buf, self.dst_buf)

    def push_to_blender(self, force=False, interval=0.0):
        if not self.is_dirty and not force:
            return

        now = time.perf_counter()
        if not force and interval > 0.0 and (now - self.last_push_time) < interval:
            return

        self.image.pixels.foreach_set(self.dst_buf.ravel())
        self.image.update()
        self.image.update_tag()

        try:
            self.image.gl_free()
            self.image.gl_load()
        except Exception:
            pass

        self.is_dirty = False
        self.last_push_time = now

    def stamp_dab(self, dst_uv, cursor_dir_uv, src_uv, src_tan_uv, props):
        w, h = self.width, self.height
        r = props.brush_radius
        strength = props.brush_strength
        hardness = props.brush_hardness

        dst_x = dst_uv.x * w
        dst_y = dst_uv.y * h
        src_x = src_uv.x * w
        src_y = src_uv.y * h

        delta_angle = math.radians(props.rake_angle_offset)

        if props.use_rake and cursor_dir_uv.length_squared > 1e-12:
            angle_cursor = math.atan2(cursor_dir_uv.y * h, cursor_dir_uv.x * w)
            if props.rake_relative_to_path and src_tan_uv.length_squared > 1e-12:
                angle_src = math.atan2(src_tan_uv.y * h, src_tan_uv.x * w)
                delta_angle += (angle_src - angle_cursor)
            else:
                delta_angle += (-angle_cursor)

        cos_a = math.cos(delta_angle)
        sin_a = math.sin(delta_angle)

        x0 = max(0, int(dst_x - r))
        x1 = min(w - 1, int(dst_x + r))
        y0 = max(0, int(dst_y - r))
        y1 = min(h - 1, int(dst_y + r))

        if x0 > x1 or y0 > y1:
            return

        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dx = xs - dst_x
        dy = ys - dst_y
        dist_sq = dx * dx + dy * dy
        mask = dist_sq <= (r * r)

        if not np.any(mask):
            return

        sx = np.rint(src_x + (dx * cos_a - dy * sin_a)).astype(np.int32)
        sy = np.rint(src_y + (dx * sin_a + dy * cos_a)).astype(np.int32)

        if props.wrap_uv:
            sx = np.mod(sx, w)
            sy = np.mod(sy, h)
        else:
            valid_src = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            mask = mask & valid_src

        if not np.any(mask):
            return

        dist_norm = np.sqrt(dist_sq[mask]) / max(1.0, float(r))
        falloff = np.where(
            dist_norm <= hardness,
            1.0,
            1.0 - (dist_norm - hardness) / max(1e-4, 1.0 - hardness)
        )
        alpha = np.clip(falloff * strength, 0.0, 1.0)[:, None]

        dy_idx = ys[mask]
        dx_idx = xs[mask]
        sy_idx = sy[mask]
        sx_idx = sx[mask]

        src_samples = self.src_buf[sy_idx, sx_idx]
        dst_samples = self.dst_buf[dy_idx, dx_idx]
        self.dst_buf[dy_idx, dx_idx] = dst_samples * (1.0 - alpha) + src_samples * alpha
        self.is_dirty = True


# =====================================================================
# GPU-PIIRTO
# =====================================================================

def draw_3d_callback():
    if not store.pts_3d:
        return
    shader = get_shader()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(3.0)
    shader.bind()

    shader.uniform_float("color", (0.0, 0.95, 1.0, 0.9))
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": store.pts_3d})
    batch.draw(shader)

    gpu.state.point_size_set(9.0)
    shader.uniform_float("color", (0.2, 1.0, 0.2, 1.0))
    batch_for_shader(shader, 'POINTS', {"pos": [store.pts_3d[0]]}).draw(shader)

    shader.uniform_float("color", (1.0, 0.5, 0.1, 1.0))
    batch_for_shader(shader, 'POINTS', {"pos": [store.pts_3d[-1]]}).draw(shader)

    if store.current_3d is not None:
        gpu.state.point_size_set(13.0)
        shader.uniform_float("color", (1.0, 0.95, 0.0, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": [store.current_3d]}).draw(shader)

        if store.brush_3d is not None:
            gpu.state.line_width_set(1.5)
            shader.uniform_float("color", (1.0, 0.95, 0.0, 0.55))
            batch_for_shader(shader, 'LINES', {"pos": [store.current_3d, store.brush_3d]}).draw(shader)

            if store.brush_dir_3d is not None:
                gpu.state.line_width_set(3.0)
                shader.uniform_float("color", (1.0, 0.4, 0.1, 0.95))
                p_end = store.brush_3d + store.brush_dir_3d * 0.15
                batch_for_shader(shader, 'LINES', {"pos": [store.brush_3d, p_end]}).draw(shader)

    gpu.state.blend_set('NONE')


def draw_2d_callback():
    context = bpy.context
    if not context.area or context.area.type != 'IMAGE_EDITOR':
        return
    if not store.uv_path:
        return

    v2d = context.region.view2d
    pts_screen = [v2d.view_to_region(uv.x, uv.y, clip=False) for uv in store.uv_path]

    shader = get_shader()
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.5)
    shader.bind()

    shader.uniform_float("color", (0.0, 0.95, 1.0, 0.9))
    batch_for_shader(shader, 'LINE_STRIP', {"pos": pts_screen}).draw(shader)

    gpu.state.point_size_set(8.0)
    shader.uniform_float("color", (0.2, 1.0, 0.2, 1.0))
    batch_for_shader(shader, 'POINTS', {"pos": [pts_screen[0]]}).draw(shader)

    shader.uniform_float("color", (1.0, 0.5, 0.1, 1.0))
    batch_for_shader(shader, 'POINTS', {"pos": [pts_screen[-1]]}).draw(shader)

    if store.current_uv is not None:
        cx, cy = v2d.view_to_region(store.current_uv.x, store.current_uv.y, clip=False)
        gpu.state.point_size_set(12.0)
        shader.uniform_float("color", (1.0, 0.95, 0.0, 1.0))
        batch_for_shader(shader, 'POINTS', {"pos": [(cx, cy)]}).draw(shader)

    gpu.state.blend_set('NONE')


def ensure_draw_handlers():
    if store.draw_handle_3d is None:
        store.draw_handle_3d = bpy.types.SpaceView3D.draw_handler_add(
            draw_3d_callback, (), 'WINDOW', 'POST_VIEW'
        )
    if store.draw_handle_2d is None:
        store.draw_handle_2d = bpy.types.SpaceImageEditor.draw_handler_add(
            draw_2d_callback, (), 'WINDOW', 'POST_PIXEL'
        )


def tag_redraw_all(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {'VIEW_3D', 'IMAGE_EDITOR'}:
                area.tag_redraw()


# =====================================================================
# OPERAATTORIT: POLUN HALLINTA (PIIRTO & POISTO)
# =====================================================================

class PAINT_OT_record_texture_path(bpy.types.Operator):
    """Piirrä tekstuurin lähdepolku suoraan 3D-objektin pinnalle tai 2D Image Editorissa"""
    bl_idname = "paint.record_ping_pong_tex_path"
    bl_label = "1. Piirrä tekstuurin lähdepolku"

    def invoke(self, context, event):
        ensure_draw_handlers()
        self.is_3d = (context.area.type == 'VIEW_3D')

        if self.is_3d:
            obj = context.active_object
            if not obj or obj.type != 'MESH':
                self.report({'WARNING'}, "Valitse ensin Mesh-objekti!")
                return {'CANCELLED'}
            if not obj.data.uv_layers.active:
                self.report({'WARNING'}, "Objektilla ei ole UV-karttaa (UV Map)!")
                return {'CANCELLED'}

        store.uv_path.clear()
        store.pts_3d.clear()
        store.current_uv = None
        store.current_3d = None
        store.brush_3d = None
        store.brush_dir_3d = None
        self._is_drawing = False

        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("LÄHDEPOLKU: Piirrä polku LMB pohjassa objektin pinnalle. Vapauta LMB valmiina. ESC peruuttaa.")
        return {'RUNNING_MODAL'}

    def add_point(self, context, event):
        if self.is_3d:
            hit, pos3d, uv = raycast_uv_on_active_mesh(context, event)
            if hit:
                store.uv_path.append(uv)
                store.pts_3d.append(pos3d)
        else:
            uv = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            store.uv_path.append(uv)

    def modal(self, context, event):
        tag_redraw_all(context)

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and not self._is_drawing:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_drawing = True
                self.add_point(context, event)
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE' and self._is_drawing:
                props = context.scene.pp_tex_clone_props
                store.uv_path, store.pts_3d = resample_path_with_3d(
                    store.uv_path, store.pts_3d, store.spacing_uv, loop=props.loop_path
                )
                context.area.header_text_set(None)
                self.report({'INFO'}, f"Tekstuurin lähdepolku tallennettu ({len(store.uv_path)} pistettä).")
                return {'FINISHED'}

        elif event.type == 'MOUSEMOVE' and self._is_drawing:
            self.add_point(context, event)
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class PAINT_OT_clear_texture_path(bpy.types.Operator):
    """Poista tallennettu lähdepolku"""
    bl_idname = "paint.clear_ping_pong_tex_path"
    bl_label = "Poista lähdepolku"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        store.uv_path.clear()
        store.pts_3d.clear()
        store.current_uv = None
        store.current_3d = None
        store.brush_3d = None
        store.brush_dir_3d = None
        tag_redraw_all(context)
        self.report({'INFO'}, "Lähdepolku poistettu.")
        return {'FINISHED'}


# =====================================================================
# OPERAATTORI 2: KLOONAUS
# =====================================================================

class PAINT_OT_clone_texture_along_path(bpy.types.Operator):
    """Maalaa tekstuuria reaaliajassa tai intervallilla lähdepolkua pitkin"""
    bl_idname = "paint.clone_ping_pong_tex_path"
    bl_label = "2. Kloonaa tekstuuria (Reaaliaikainen)"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        ensure_draw_handlers()
        if len(store.uv_path) < 2:
            self.report({'WARNING'}, "Piirrä ensin lähdepolku (Vaihe 1)!")
            return {'CANCELLED'}

        self.img = find_target_image(context)
        if not self.img:
            self.report({'WARNING'}, "Ei löytynyt muokattavaa tekstuurikuvaa!")
            return {'CANCELLED'}

        self.is_3d = (context.area.type == 'VIEW_3D')
        self.cloner = FastPixelCloner(self.img)

        self._is_painting = False
        self.dist_accum_uv = 0.0
        self.last_dst_uv = None
        self.last_pos_3d = None
        self.smoothed_cursor_dir_uv = Vector((1.0, 0.0))

        props = context.scene.pp_tex_clone_props
        timer_step = max(0.005, props.update_interval)
        self._timer = context.window_manager.event_timer_add(timer_step, window=context.window)

        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(
            f"REAALIAIKAINEN KLOONAUS ({self.img.name}): Pidä LMB pohjassa, MMB kääntää näkymää, RMB / ESC lopettaa."
        )
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        store.current_uv = None
        store.current_3d = None
        store.brush_3d = None
        store.brush_dir_3d = None
        context.area.header_text_set(None)
        tag_redraw_all(context)

    def get_mouse_hit(self, context, event):
        if self.is_3d:
            hit, pos3d, uv = raycast_uv_on_active_mesh(context, event)
            if hit:
                if self.last_pos_3d is not None and (pos3d - self.last_pos_3d).length > 1e-4:
                    raw_dir_3d = (pos3d - self.last_pos_3d).normalized()
                    if store.brush_dir_3d is None or not store.current_3d:
                        store.brush_dir_3d = raw_dir_3d
                    else:
                        smooth = context.scene.pp_tex_clone_props.rake_smoothing
                        store.brush_dir_3d = store.brush_dir_3d.lerp(raw_dir_3d, 1.0 - smooth).normalized()
                self.last_pos_3d = pos3d
                store.brush_3d = pos3d
                return uv
            store.brush_3d = None
            return None
        else:
            return Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))

    def paint_step_to(self, cur_uv, props):
        max_dim = max(1, max(self.cloner.width, self.cloner.height))
        step_uv = max(0.0005, (props.brush_radius * props.dab_spacing) / max_dim)

        if self.last_dst_uv is None:
            self.last_dst_uv = cur_uv
            src_uv, src_tan, src_3d = sample_path_mode(
                store.uv_path, store.pts_3d, self.dist_accum_uv, store.spacing_uv, loop=props.loop_path
            )
            store.current_uv = src_uv
            store.current_3d = src_3d
            self.smoothed_cursor_dir_uv = src_tan.copy()
            self.cloner.stamp_dab(cur_uv, self.smoothed_cursor_dir_uv, src_uv, src_tan, props)
            self.cloner.push_to_blender(force=True)
            return

        delta_uv = cur_uv - self.last_dst_uv

        if delta_uv.length > 0.25:
            self.last_dst_uv = cur_uv
            return

        if delta_uv.length >= step_uv:
            raw_dir = delta_uv.normalized()
            smooth_factor = 1.0 - min(0.95, props.rake_smoothing)
            self.smoothed_cursor_dir_uv = self.smoothed_cursor_dir_uv.lerp(raw_dir, smooth_factor)
            if self.smoothed_cursor_dir_uv.length_squared > 1e-12:
                self.smoothed_cursor_dir_uv.normalize()
            else:
                self.smoothed_cursor_dir_uv = raw_dir

            steps = max(1, int(delta_uv.length / step_uv))
            for i in range(1, steps + 1):
                interp_uv = self.last_dst_uv.lerp(cur_uv, i / steps)
                self.dist_accum_uv += step_uv * props.ping_pong_speed

                src_uv, src_tan, src_3d = sample_path_mode(
                    store.uv_path, store.pts_3d, self.dist_accum_uv, store.spacing_uv, loop=props.loop_path
                )
                store.current_uv = src_uv
                store.current_3d = src_3d
                self.cloner.stamp_dab(interp_uv, self.smoothed_cursor_dir_uv, src_uv, src_tan, props)

            self.last_dst_uv = cur_uv
            self.cloner.push_to_blender(force=False, interval=props.update_interval)

    def modal(self, context, event):
        props = context.scene.pp_tex_clone_props

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and not self._is_painting:
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            if self._is_painting:
                if props.continuous_interval_feed and self.last_dst_uv is not None:
                    max_dim = max(1, max(self.cloner.width, self.cloner.height))
                    step_uv = max(0.0005, (props.brush_radius * props.dab_spacing) / max_dim)
                    self.dist_accum_uv += step_uv * props.ping_pong_speed
                    src_uv, src_tan, src_3d = sample_path_mode(
                        store.uv_path, store.pts_3d, self.dist_accum_uv, store.spacing_uv, loop=props.loop_path
                    )
                    store.current_uv = src_uv
                    store.current_3d = src_3d
                    self.cloner.stamp_dab(self.last_dst_uv, self.smoothed_cursor_dir_uv, src_uv, src_tan, props)

                if self.cloner.is_dirty:
                    self.cloner.push_to_blender(force=False, interval=props.update_interval)
                    tag_redraw_all(context)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.last_pos_3d = None
                cur_uv = self.get_mouse_hit(context, event)
                if cur_uv is None:
                    return {'RUNNING_MODAL'}

                self._is_painting = True
                if props.reset_each_stroke:
                    self.dist_accum_uv = 0.0
                self.last_dst_uv = None
                self.paint_step_to(cur_uv, props)
                tag_redraw_all(context)
                return {'RUNNING_MODAL'}

            elif event.value == 'RELEASE' and self._is_painting:
                self._is_painting = False
                self.last_dst_uv = None
                self.last_pos_3d = None
                store.brush_dir_3d = None
                self.cloner.push_to_blender(force=True)
                self.cloner.commit_stroke_to_source()
                tag_redraw_all(context)
                return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE':
            cur_uv = self.get_mouse_hit(context, event)
            if self._is_painting and cur_uv is not None:
                self.paint_step_to(cur_uv, props)
            tag_redraw_all(context)
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cleanup(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}


# =====================================================================
# ASETUKSET JA UI-PANEELIT
# =====================================================================

class PingPongTexCloneProperties(bpy.types.PropertyGroup):
    target_image: bpy.props.PointerProperty(
        name="Tekstuuri",
        type=bpy.types.Image,
        description="Valitse muokattava tekstuurikuva"
    )
    brush_radius: bpy.props.IntProperty(
        name="Säde (px)", default=25, min=2, max=250,
        description="Kloonaussiveltimen säde tekstuurin pikseleinä"
    )
    brush_strength: bpy.props.FloatProperty(
        name="Voimakkuus", default=1.0, min=0.05, max=1.0, subtype='FACTOR',
        description="Siveltimen peittävyys"
    )
    brush_hardness: bpy.props.FloatProperty(
        name="Kovuus", default=0.65, min=0.0, max=1.0, subtype='FACTOR',
        description="Siveltimen reunan terävyys"
    )
    dab_spacing: bpy.props.FloatProperty(
        name="Leimausväli (Spacing)", default=0.20, min=0.05, max=1.0, subtype='FACTOR',
        description="Kuinka tiheästi sivellin leimaa liikkuessaan"
    )
    ping_pong_speed: bpy.props.FloatProperty(
        name="Nopeus", default=1.0, min=0.1, max=5.0,
        description="Kuinka nopeasti lukupiste etenee lähdepolulla"
    )
    loop_path: bpy.props.BoolProperty(
        name="Sulje polku (Loop)", default=False,
        description="Jos päällä, polku muodostaa suljetun silmukan ja kloonaus kiertää rataa jatkuvasti ympäri samaan suuntaan"
    )
    update_interval: bpy.props.FloatProperty(
        name="Päivitysväli (s)", default=0.02, min=0.005, max=0.5, step=1, precision=3,
        description="Kuinka usein tekstuuri päivitetään näytönohjaimelle"
    )
    continuous_interval_feed: bpy.props.BoolProperty(
        name="Jatkuva syöttö paikallaan (Airbrush)", default=False,
        description="Jos päällä, polku etenee ja maalaa myös silloin, kun hiiri on paikallaan"
    )

    use_rake: bpy.props.BoolProperty(
        name="Rake (Kierrä liikesuunnan mukaan)", default=True,
        description="Kytkee päälle siveltimen kulman kierron kursorin liikesuunnan mukaan"
    )
    rake_relative_to_path: bpy.props.BoolProperty(
        name="Suhteuta lähdepolun suuntaan", default=True,
        description="Kääntää lähdepolun tangentin kursorin liikesuunnan mukaiseksi"
    )
    rake_smoothing: bpy.props.FloatProperty(
        name="Rake-pehmennys", default=0.35, min=0.0, max=0.95, subtype='FACTOR',
        description="Tasoittaa kursorin liikesuunnan muutoksia"
    )
    rake_angle_offset: bpy.props.FloatProperty(
        name="Kulman poikkeama (°)", default=0.0, min=-180.0, max=180.0,
        description="Kiinteä lisäkulma asteina siveltimen kiertoon"
    )

    reset_each_stroke: bpy.props.BoolProperty(
        name="Aloita alusta joka vedolla", default=False,
        description="Aloittaa jokaisen uuden siveltimen vedon lähdepolun alkupisteestä"
    )
    wrap_uv: bpy.props.BoolProperty(
        name="Saumaton UV-kierto (Tile/Wrap)", default=True,
        description="Kiertää tekstuurin reunojen yli"
    )


def draw_clone_ui(layout, context):
    props = context.scene.pp_tex_clone_props
    img = find_target_image(context)

    box_img = layout.box()
    box_img.label(text="Kohdetekstuuri:", icon='IMAGE_DATA')
    box_img.prop(props, "target_image", text="")
    if img:
        box_img.label(text=f"Aktiivinen: {img.name} ({img.size[0]}x{img.size[1]})", icon='CHECKMARK')
    else:
        box_img.label(text="Ei tekstuuria! Valitse yltä.", icon='ERROR')

    layout.separator()

    col = layout.column(align=True)
    col.scale_y = 1.35
    col.operator("paint.record_ping_pong_tex_path", icon='CURVE_PATH')
    
    row_path = col.row(align=True)
    row_path.operator("paint.clone_ping_pong_tex_path", icon='BRUSH_DATA')
    if store.uv_path:
        row_path.operator("paint.clear_ping_pong_tex_path", icon='X', text="")

    if store.uv_path:
        layout.label(text=f"Polussa pisteitä: {len(store.uv_path)}", icon='INFO')

    layout.separator()
    box_loop = layout.box()
    box_loop.prop(props, "loop_path", icon='FILE_TICK')

    layout.separator()
    box_rake = layout.box()
    row = box_rake.row()
    row.prop(props, "use_rake", icon='DRIVER_ROTATIONAL_DIFFERENCE')
    if props.use_rake:
        sub = box_rake.column(align=True)
        sub.prop(props, "rake_relative_to_path")
        sub.prop(props, "rake_smoothing")
        sub.prop(props, "rake_angle_offset")
    else:
        box_rake.prop(props, "rake_angle_offset", text="Kiinteä kulma (°)")

    layout.separator()
    box_rt = layout.box()
    box_rt.label(text="Reaaliaika & Intervalli:", icon='TIME')
    box_rt.prop(props, "update_interval")
    box_rt.prop(props, "dab_spacing")
    box_rt.prop(props, "continuous_interval_feed")

    layout.separator()
    box = layout.box()
    box.label(text="Siveltimen asetukset:", icon='PREFERENCES')
    box.prop(props, "brush_radius")
    box.prop(props, "brush_strength")
    box.prop(props, "brush_hardness")
    box.prop(props, "ping_pong_speed")
    box.prop(props, "reset_each_stroke")
    box.prop(props, "wrap_uv")


class VIEW3D_PT_ping_pong_tex_clone(bpy.types.Panel):
    bl_label = "Ping-Pong Texture Clone"
    bl_idname = "VIEW3D_PT_ping_pong_tex_clone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Path Clone"

    def draw(self, context):
        draw_clone_ui(self.layout, context)


class IMAGE_PT_ping_pong_tex_clone(bpy.types.Panel):
    bl_label = "Ping-Pong Texture Clone"
    bl_idname = "IMAGE_PT_ping_pong_tex_clone"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Path Clone"

    def draw(self, context):
        draw_clone_ui(self.layout, context)


classes = (
    PingPongTexCloneProperties,
    PAINT_OT_record_texture_path,
    PAINT_OT_clear_texture_path,
    PAINT_OT_clone_texture_along_path,
    VIEW3D_PT_ping_pong_tex_clone,
    IMAGE_PT_ping_pong_tex_clone,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pp_tex_clone_props = bpy.props.PointerProperty(type=PingPongTexCloneProperties)


def unregister():
    if store.draw_handle_3d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(store.draw_handle_3d, 'WINDOW')
        store.draw_handle_3d = None
    if store.draw_handle_2d is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(store.draw_handle_2d, 'WINDOW')
        store.draw_handle_2d = None

    del bpy.types.Scene.pp_tex_clone_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()