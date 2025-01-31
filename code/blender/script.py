import os
import bpy
import glob

#bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.delete(use_global=False)
#bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


BASE_PATH = '<Path to Merged Image Tiles>'

def remove_sky():
    world = bpy.context.scene.world

    if world and world.use_nodes:
        nodes = world.node_tree.nodes
        nodes.clear()
        bg_node = nodes.new(type="ShaderNodeBackground")
        output_node = nodes.new(type="ShaderNodeOutputWorld")
        bg_node.location = (0, 0)
        output_node.location = (300, 0)
        links = world.node_tree.links
        links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])
        bg_node.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)  # RGBA

    bpy.context.view_layer.update()

def add_sky():
    remove_sky()
    world = bpy.data.worlds.get("World")
    if not world:
        world = bpy.data.worlds.new("World")

    bpy.context.scene.world = world

    world.use_nodes = True

    nodes = world.node_tree.nodes
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.location = (0, 0)
    sky_node = nodes.new(type="ShaderNodeTexSky")
    sky_node.location = (-300, 0)
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (300, 0)

    links = world.node_tree.links
    links.new(sky_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    sky_node.sky_type = 'NISHITA'
    sky_node.altitude = 1500
    bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[1].default_value = 0.3

    # Update the viewport
    bpy.context.view_layer.update()

def load_base_image(east, north, size):
#    satellite_image_path = f"/Users/felix/MSE/VT1/01_data/dataset/base/{east}_{north}.png"
#    satellite_image_path = f"/Users/felix/MSE/VT1/03_mae/out/{east}_{north}.png"

    satellite_image_path = f"{BASE_PATH}/out/{east}_{north}_{size}.png"
    
    if os.path.isfile(satellite_image_path):
        return bpy.data.images.load(satellite_image_path)
        

def load_alit(east, north, size):
    alti_image_path = f"{BASE_PATH}/alti/{east}_{north}_{size}.png"
    
    if os.path.isfile(alti_image_path):
        return bpy.data.images.load(alti_image_path)

def load_tile(east, north, size):
    bpy.ops.mesh.primitive_plane_add(size=2*size, location=(0,0,0))
    plane = bpy.context.object

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=1000000000)
    bpy.ops.object.mode_set(mode='OBJECT')

    displace_modifier = plane.modifiers.new(name="Displace", type='DISPLACE')
    
    texture = bpy.data.textures.new("ElevationMap", type='IMAGE')
    texture.crop_min_x = 0.01
    texture.crop_min_y = 0.01
    texture.crop_max_x = 0.99
    texture.crop_max_y = 0.99
    texture.image = load_alit(east, north, size)
    
    displace_modifier.texture = texture
    displace_modifier.strength = 35
    displace_modifier.texture_coords = 'UV'

    material = bpy.data.materials.new(name="SatelliteMaterial")
    material.use_nodes = True
    plane.data.materials.append(material)

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in nodes:
        nodes.remove(node)
        
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader_node.location = (200, 0)

    image_node = nodes.new(type='ShaderNodeTexImage')
    image_node.location = (0, 0)

    image_node.image = load_base_image(east, north, size)
    
    links.new(image_node.outputs['Color'], shader_node.inputs['Base Color'])
    links.new(shader_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    
    
east = 2620
north = 1095
size = 25

#east = 2620
#north = 1095
#size = 25

load_tile(east, north, size)

#remove_sky()
#add_sky()

#bpy.ops.object.camera_add(location=(14, 14, 100))
#camera = bpy.context.object
#camera.type = 'ORTHO'
#camera.ortho_scale = 30

#camera.rotation_euler = (78, 0, 135)  # Adjust angle as needed
#bpy.context.scene.camera = camera

bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
light = bpy.context.object
#light.data.energy = 5