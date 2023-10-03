from array import array
import struct
import numpy as np
import os

class_colors = [
    #(0.0, 0.0, 0.0),
    (0.1, 0.1, 0.1),
    (0.0649613, 0.467197, 0.0667303),
    (0.1, 0.847035, 0.1),
    (0.0644802, 0.646941, 0.774265),
    (0.131518, 0.273524, 0.548847),
    (1, 0.813553, 0.0392201),
    (1, 0.490452, 0.0624932),
    (0.657877, 0.0505005, 1),
    (0.0363214, 0.0959549, 0.548847),
    (0.316852, 0.548847, 0.186899),
    (0.548847, 0.143381, 0.0045568),
    (1, 0.241096, 0.718126),
    (0.9, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.3, 0.3, 0.3)
    ]
class_names = ["empty", "ceiling", "floor", "wall", "window", "chair", "bed", "sofa",
               "table", "tvs", "furniture", "objects", "error1", "error2", "error3"]

def voxel_export(name, vox):

    shape = vox.shape

    coord_x=array('i')
    coord_y=array('i')
    coord_z=array('i')
    voxel_v=array('f')

    count = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                coord_x.append(x)
                coord_y.append(y)
                coord_z.append(z)
                voxel_v.append(vox[x,y,z].item()) 
                count += 1

    print("saving...")
    with open(name, 'wb') as f:
        f.write(struct.pack("i", count))
        f.write(struct.pack(str(count)+"i", *coord_x))
        f.write(struct.pack(str(count)+"i", *coord_y))
        f.write(struct.pack(str(count)+"i", *coord_z))
        f.write(struct.pack(str(count)+"f", *voxel_v))
    
    print(count, "done...")
   
               
 
def write_header(obj_file, mtl_file, name):
    obj_file.write("# R-Unet Wavefront obj exporter v1.0\n")
    obj_file.write("mtllib %s.mtl\n" % os.path.basename(name))
    obj_file.write("o Cube\n")

    mtl_file.write("# R-Unet Wavefront obj exporter v1.0\n")
    # Blender MTL File: 'DWRC1.blend'
    # Material Count: 11


def write_vertice(obj_file, x, y, z, cx, cy, cz, v_unit):
    vu = v_unit * 4
    obj_file.write("v %8.6f %8.6f %8.6f\n" %((x-cx)*vu, (y-cy)*vu, (z-cz)*vu))

def write_vertice_normals(obj_file): # normals are associated with the vertices of the cube to provide the correct shading and lighting information for each face
    obj_file.write("vn -1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000 -1.000000\n")
    obj_file.write("vn  1.000000  0.000000  0.000000\n")
    obj_file.write("vn  0.000000  0.000000  1.000000\n")
    obj_file.write("vn  0.000000 -1.000000  0.000000\n")
    obj_file.write("vn  0.000000  1.000000  0.000000\n")


def write_mtl_faces(obj_file, mtl_file, mtl_faces_list, cl, triangular): #define the material properties of the 3D object and are essential for achieving realistic appearance and shading in 3D modeling software or game engines
    obj_file.write("g %s\n" % class_names[cl])
    obj_file.write("usemtl %s\n" % class_names[cl])
    mtl_file.write("newmtl %s\n" % class_names[cl])
    mtl_file.write("Ns 96.078431\n")
    mtl_file.write("Ka 1.000000 1.000000 1.000000\n")
    mtl_file.write("Kd %8.6f %8.6f %8.6f\n" % (class_colors[cl][0], class_colors[cl][1], class_colors[cl][2] ) )
    mtl_file.write("Ks 0.500000 0.500000 0.500000\n")
    mtl_file.write("Ke 0.000000 0.000000 0.000000\n")
    mtl_file.write("Ni 1.000000\n")
    mtl_file.write("d 1.000000\n")
    mtl_file.write("illum 2\n")


    if not triangular:
        for face_vertices in mtl_faces_list:

            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d %d//%d" % (
                                                                face_vertices[1]+1, face_vertices[0],
                                                                face_vertices[2]+1, face_vertices[0],
                                                                face_vertices[3]+1, face_vertices[0],
                                                                face_vertices[4]+1, face_vertices[0],
                                                            ))
            obj_file.write("\n")
    else:
        for face_vertices in mtl_faces_list:
            obj_file.write("f ")
            obj_file.write("%d//%d  %d//%d %d//%d" % (
                                                            face_vertices[1]+1, face_vertices[0],
                                                            face_vertices[2]+1, face_vertices[0],
                                                            face_vertices[3]+1, face_vertices[0],
                                                        ))
            obj_file.write("\n")


def obj_export(name, vox, shape, camx  , camy, camz , v_unit, include_top=False, triangular=False, inner_faces=True):

    gap = 0.1

    _vox = np.swapaxes(vox, axis1=0,axis2=2)
    
    num_classes=len(class_names)

    sx, sy, sz = _vox.shape
    
    cx, cy, cz = sx, sy, sz
    vox_ctrl = np.ones((sx+1, sy+1, sz+1), dtype=np.int32) 
    
    mtl_faces_list =[None] * num_classes

    num_vertices = 0

    with open(name+".obj", 'w') as obj_file, open(name+".mtl", 'w') as mtl_file:

        write_header(obj_file, mtl_file, name)

        for x in range(sx):
            for y in range(sy):
                if not include_top and y>26:
                        continue
                for z in range(sz):
                    mtl = int(_vox[x,y,z])
                    if mtl == 0 or mtl==255:
                        continue
                    delta = [gap/2, 1-gap/2]
                    
                    for vx in range(2):
                        for vy in range(2):
                            for vz in range(2):
                                vox_ctrl[x + vx, y + vy, z + vz] = num_vertices
                                num_vertices += 1
                                write_vertice(obj_file, x+delta[vx], y+delta[vy], z+delta[vz], cx, cy, cz, v_unit)
                    if mtl_faces_list[mtl] is None:
                        mtl_faces_list[mtl] = []

                    if inner_faces:

                        if triangular:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+1]], #OK
                                [1, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+0, y+0, z+1]], #OK

                                [2, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+0, y+0, z+0]], #OK
                                [2, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+0, y+0, z+0]], #OK

                                [3, vox_ctrl[x+1, y+1, z+0], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+0]], #OK
                                [3, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+0, z+1], vox_ctrl[x+1, y+0, z+0]], #OK

                                [4, vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+0, z+1]], #OK
                                [4, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+1, y+0, z+1]], #OK

                                [5, vox_ctrl[x+0, y+0, z+1], vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK
                                [5, vox_ctrl[x+0, y+0, z+0], vox_ctrl[x+1, y+0, z+0], vox_ctrl[x+1, y+0, z+1]], #OK

                                [6, vox_ctrl[x+0, y+1, z+0], vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+0]], #OK
                                [6, vox_ctrl[x+0, y+1, z+1], vox_ctrl[x+1, y+1, z+1], vox_ctrl[x+1, y+1, z+0]]  #OK
                            ])

                        else:

                            mtl_faces_list[mtl].extend([
                                [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                 vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                 vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                 vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]  # OK
                            ])
                    else:
                        _x, _y, _z = x+1, y+1, z+1
                        if triangular:

                            if _vox[_x - 1,_y,_z] != _vox[_x, _y, _z] and _vox[_x-1,_y,_z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]],  # OK
                                    [1, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y, _z-1] != _vox[_x, _y, _z] and _vox[_x, _y, _z-1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]],  # OK
                                    [2, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0]]])  # OK

                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x + 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]],  # OK
                                    [3, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 0]]])  # OK

                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z + 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [4, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 0, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y-1, _z] != _vox[_x, _y, _z] and _vox[_x, _y-1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]],  # OK
                                    [5, vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 1]]])  # OK

                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y + 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]],  # OK
                                    [6, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

                        else:

                            if _vox[_x - 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x - 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [1, vox_ctrl[x + 0, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 0],
                                     vox_ctrl[x + 0, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y, _z - 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z - 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [2, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 0, y + 0, z + 0]]])  # OK
                            if _vox[_x + 1, _y, _z] != _vox[_x, _y, _z] and _vox[_x + 1, _y, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [3, vox_ctrl[x + 1, y + 1, z + 0], vox_ctrl[x + 1, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 0]]])  # OK
                            if _vox[_x, _y, _z + 1] != _vox[_x, _y, _z] and _vox[_x, _y, _z + 1]!=255:
                                mtl_faces_list[mtl].extend([
                                    [4, vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y - 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y - 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [5, vox_ctrl[x + 0, y + 0, z + 1], vox_ctrl[x + 0, y + 0, z + 0],
                                     vox_ctrl[x + 1, y + 0, z + 0], vox_ctrl[x + 1, y + 0, z + 1]]])  # OK
                            if _vox[_x, _y + 1, _z] != _vox[_x, _y, _z] and _vox[_x, _y + 1, _z]!=255:
                                mtl_faces_list[mtl].extend([
                                    [6, vox_ctrl[x + 0, y + 1, z + 0], vox_ctrl[x + 0, y + 1, z + 1],
                                     vox_ctrl[x + 1, y + 1, z + 1], vox_ctrl[x + 1, y + 1, z + 0]]])  # OK

        write_vertice_normals(obj_file)

        for mtl in range(num_classes):
            if not  mtl_faces_list[mtl] is None:
                write_mtl_faces(obj_file, mtl_file, mtl_faces_list[mtl], mtl, triangular)
    
    print(f"Obj export done")
    return
    
