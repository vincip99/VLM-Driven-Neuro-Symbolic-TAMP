from robosuite.models.arenas import TableArena
import xml.etree.ElementTree as ET

class CustomArena(TableArena):
    """
    Custom arena class to modify lighting and cameras
    """
    def __init__(
            self,
            table_full_size=(1.4, 0.8, 0.05),
            table_offset=(0, 0, 0.8),
            has_legs=False
        ):
        # Initialize inherit class TableArena from Robosuite/models/arenas
        super().__init__(
            table_full_size=table_full_size,
            table_offset=table_offset,
            has_legs=has_legs
        )

        # add camera and light properties
        self._set_shadowless_lighting()
        self._add_cameras()

    def _set_shadowless_lighting(self):
        """Removes harsh default lights and adds diffuse ambient lighting."""
        for light in self.worldbody.findall("./light"):
            self.worldbody.remove(light)
            
        ambient_light = ET.Element(
                "light",
                name="vlm_ambient",
                dir="0 0 -1",
                pos="0 0 3.0",
                castshadow="false", 
                diffuse="0.8 0.8 0.8",
                specular="0.1 0.1 0.1",
            )
        self.worldbody.append(ambient_light)

    def _add_cameras(self):
        """Adds both the VLM top-down camera and the human debug camera."""
        # Top-Down VLM Camera (Placed 1.35 meters above the table, centered over workspace)
        cam_height = self.table_offset[2] + 1.35
        top_down_cam = ET.Element(
            "camera",
            name="top_down_vlm",
            pos=f"-0.05 -0.10 {cam_height}",
            mode="fixed",
            quat="1 0 0 0" 
        )
        self.worldbody.append(top_down_cam)

        # Frontal Debug Camera (Placed in front of the table, angled down)
        frontal_cam = ET.Element(
            "camera",
            name="frontal_debug",
            pos="1.2 0 1.2", 
            mode="fixed",
            quat="0.560842 0.430459 0.430459 0.560842" 
        )
        self.worldbody.append(frontal_cam)

        # Isometric 3D Overview Cameras (angled down at arena center)
        iso_cam_left = ET.Element(
            "camera",
            name="arena_isometric_left",
            pos="1.15 -0.85 1.45",
            mode="fixed",
            quat="0.743514 0.494350 0.249339 0.375012"
        )
        self.worldbody.append(iso_cam_left)

        iso_cam_right = ET.Element(
            "camera",
            name="arena_isometric_right",
            pos="1.2 0.75 1.45",
            mode="fixed",
            quat="0.404188 0.267643 0.482895 0.729255"
        )
        self.worldbody.append(iso_cam_right)