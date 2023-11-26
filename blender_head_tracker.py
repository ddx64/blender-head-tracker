import bpy
import cv2
import numpy as np

# ----------Meta Info----------

bl_info = {
    "name": "Gaze Navi",
    "description": "OpenCV based camera navigator",
    "author": "ddx64",
    "version": (1, 0),
    "blender": (3, 6, 2),
    "location": "View3D > Panel > Tool",
    "doc_url": "https://github.com/ddx64/blender-head-tracker.git",
    "category": "Generic"
}

# ----------Meta Info----------


# ----------Main operators----------

class GAZE_OT_Zoom(bpy.types.Operator):
    bl_idname = "gaze.zoom"
    bl_label = "Gaze Navi Zoom"

    def __init__(self) -> None:
        global snapshot
        self.frame_generator = snapshot()
        self.buffer = None

    @staticmethod
    def face_filter(frame):
        global face_classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(
            gray, 1.1, 5, minSize=(100, 100), maxSize=(250, 250))
        if len(face) != 1:
            return None
        fx, fy, fw, fh = face[0]
        return fw

    @classmethod
    def poll(cls, context):
        return context.scene.RNAswitch

    def execute(self, context):
        print("gaze zoom executed")
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type != 'T' and 'MOUSE' not in event.type:
            self.buffer = None
            return {"FINISHED"}

        fsize, cnt = 0, 0
        for _ in range(5):
            k = GAZE_OT_Zoom.face_filter(next(self.frame_generator))
            if k is not None:
                fsize += k
                cnt += 1
        if cnt == 0:
            return {'RUNNING_MODAL'}
        fsize = round(fsize/cnt/3)
        # print(f"size {fsize}")

        if self.buffer is None:
            self.buffer = fsize
            return {'RUNNING_MODAL'}

        self.rv3d.view_distance += (self.buffer-fsize)*context.scene.RNAefact
        self.buffer = fsize

        print("gaze zoom executed")
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.rv3d = None

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                self.rv3d = area.spaces[0].region_3d

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

        # self.execute(context)
        # return {'FINISHED'}


class GAZE_OT_Rotate(bpy.types.Operator):
    bl_idname = "gaze.rotate"
    bl_label = "Gaze Navi Rotate"

    buffer_size = 5

    def __init__(self) -> None:
        global snapshot
        self.frame_generator = snapshot()
        self.buffer_x = []
        self.buffer_y = []
        self.last_offset = (0, 0)

    # input -> <frame>
    # output <- [<subframe>,(xywh)]*2

    @staticmethod
    def eyes_detect(frame):
        global face_classifier, eye_classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(
            gray, 1.1, 5, minSize=(50, 70))
        if len(face) != 1:
            return None, None
        eyes = eye_classifier.detectMultiScale(gray, 1.2, 10)

        left, right = None, None
        fx, fy, fw, fh = face[0]
        for x, y, w, h in eyes:
            if x < fx or x+w > fx+fw or y < fy or y+h > fy+fh:
                continue
            if x < fx+fw/2:
                left = (gray[y:y+h, x:x+h], (x, y, w, h))
            else:
                right = (gray[y:y+h, x:x+h], (x, y, w, h))
        return left, right

    # input -> <subframe>
    # output <- (xywh)*2

    @staticmethod
    def pupil_filter(frame):
        pupil = cv2.HoughCircles(
            frame, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=13, minRadius=7, maxRadius=20)
        if pupil is None:
            return None
        return np.round(pupil[0, 0, :2]).astype("int")

    # input -> (xywh)*4
    # output <- (xywh)*2

    @staticmethod
    def mux(left, right, left_pupil, right_pupil) -> tuple:
        if left_pupil is None:
            return (right_pupil[0]/right[2], right_pupil[1]/right[3])
        if right_pupil is None:
            return (left_pupil[0]/left[2], left_pupil[1]/left[3])
        return (right_pupil[0]+left_pupil[0])/(right[2]+left[2]), \
            (right_pupil[1]+right_pupil[1])/(right[3]+left[3])

    # return changes compare to last offset

    def compare(self, x, y):
        if x is None or y is None:
            return (0, 0)
        sx, sy = self.last_offset
        res = (x-sx, y-sy)
        self.lastoffset = (x, y)
        return res

    # calaulate pupil offset, based on (50%,50%)
    def pupil_offset(self, arg_x, arg_y) -> tuple:
        factor_x, factor_y = 16, 9
        if len(self.buffer_x) >= GAZE_OT_Rotate.buffer_size:
            del self.buffer_x[0]
            del self.buffer_y[0]
        self.buffer_x.append(arg_x)
        self.buffer_y.append(arg_y)
        x = sum(self.buffer_x)/GAZE_OT_Rotate.buffer_size
        y = sum(self.buffer_y)/GAZE_OT_Rotate.buffer_size
        return np.round((x-0.5)*factor_x).astype('int'), \
            np.round((y-0.5)*factor_y).astype('int')

    # output final offset of one frame
    def scan(self):
        frame = next(self.frame_generator)
        left, right = GAZE_OT_Rotate.eyes_detect(frame)
        left_args, right_args = None, None
        left_pupil, right_pupil = None, None
        if left:
            left_eye, left_args = left
            left_pupil = GAZE_OT_Rotate.pupil_filter(left_eye)
        if right:
            right_eye, right_args = right
            right_pupil = GAZE_OT_Rotate.pupil_filter(right_eye)
        if left_pupil is None and right_pupil is None:
            return None, None
        mx, my = GAZE_OT_Rotate.mux(
            left_args, right_args, left_pupil, right_pupil)
        ox, oy = self.pupil_offset(mx, my)
        # print(f"offset {ox} {oy}")
        return ox, oy

    @classmethod
    def poll(cls, context):
        return context.scene.RNAswitch

    # ignore this

    def execute(self, context):
        print("gaze rotate executed")
        return {'FINISHED'}

    def modal(self, context, event):
        # print(event.type)
        if event.type != 'R' and 'MOUSE' not in event.type:
            return {"FINISHED"}
        e_fact = context.scene.RNAefact
        ox, oy = self.scan()
        x, y = self.compare(ox, oy)
        ## Rotate Viewport ##
        if self.rv3d is not None:
            # self.rv3d.view_location.x += 1.0
            # self.rv3d.view_location.y += 1.0
            if x > 0:
                bpy.ops.view3d.view_orbit(
                    context.scene.RNAefact, type="ORBITRIGHT")
            elif x < 0:
                bpy.ops.view3d.view_orbit(
                    -context.scene.RNAefact, type="ORBITLEFT")
            # bpy.ops.view3d.view_pan(self.rv3d, type="PANUP")
        ## Rotate Viewport ##
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        self.rv3d = None

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                # self.rv3d = area.spaces[0].region_3d
                self.rv3d = {'area': area, 'region': area.regions[-1]}

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

        # self.execute(context)
        # return {"FINISHED"}

# ----------Main operators----------


# ----------Panel layout----------

class NAVI_PT_view3d(bpy.types.Panel):
    bl_idname = "NAVI_PT_view3d"
    bl_label = "Gaze Navi"
    ## UI Location ##
    bl_category = 'Tool'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode"
    ## UI Location ##

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = layout.column()
        row1 = col.row()
        row1.label(icon='BLENDER')
        row1.prop(scene, "RNAswitch", text='activate')
        row2 = col.row()
        row2.label(icon='TRACKER')
        row2.prop(scene, "RNAefact", text='sensitive')
        row3 = col.row()
        row3.label(text="Rotate")
        row3.label(text="SHIFT + R")
        row4 = col.row()
        row4.label(text="Zoom")
        row4.label(text="SHIFT + T")

# ----------Panel layout----------


# ----------Mount Plugin----------

def register():
    ## Register Classes ##
    bpy.utils.register_class(NAVI_PT_view3d)
    bpy.utils.register_class(GAZE_OT_Rotate)
    bpy.utils.register_class(GAZE_OT_Zoom)
    ## Register Classes ##

    ## Setup RNA param ##
    bpy.types.Scene.RNAswitch = bpy.props.BoolProperty("RNAswitch")
    bpy.types.Scene.RNAefact = bpy.props.IntProperty(
        "RNAefact", default=1, min=0, max=10, step=1)
    ## Setup RNA param ##

    ## Setup OpenCV param ##
    global face_classifier, eye_classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    eye_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_eye.xml")
    ## Setup OpenCV param ##

    ## Setup CV Generator ##
    global CV_CAMERA, snapshot

    CV_CAMERA = cv2.VideoCapture(0)

    def snapshot():
        while True:
            retval, image = CV_CAMERA.read()
            if retval:
                yield image
    ## Setup CV Generator ##

    ## Generate KeyMap ##
    global addon_keymaps

    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(
        name="3D View Generic", space_type="VIEW_3D")
    kmr = km.keymap_items.new(GAZE_OT_Rotate.bl_idname,
                              "R", "PRESS", shift=True)
    kmz = km.keymap_items.new(GAZE_OT_Zoom.bl_idname,
                              "T", "PRESS", shift=True)
    addon_keymaps = []
    addon_keymaps.append((km, kmr))
    addon_keymaps.append((km, kmz))
    ## Generate KeyMap ##

# ----------Mount Plugin----------


# ----------UnMount Plugin----------

def unregister():
    ## Unregister Classes ##
    bpy.utils.unregister_class(NAVI_PT_view3d)
    bpy.utils.unregister_class(GAZE_OT_Rotate)
    bpy.utils.unregister_class(GAZE_OT_Zoom)
    ## Unregister Classes ##

    ## Remove RNA param ##
    del bpy.types.Scene.RNAswitch
    del bpy.types.Scene.RNAefact
    ## Remove RNA param ##

    ## Remove OpenCV param ##
    global face_classifier, eye_classifier
    del face_classifier
    del eye_classifier
    ## Remove OpenCV param ##

    ## Remove CV Generator ##
    global CV_CAMERA, snapshot
    del CV_CAMERA
    del snapshot
    ## Remove CV Generator ##

    ## Remove KeyMap ##
    global addon_keymaps

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    del addon_keymaps
    ## Remove KeyMap ##

# ----------UnMount Plugin----------

# ----------script test only---------
# if __name__ == "__main__":
#     register()
# ----------script test only---------
