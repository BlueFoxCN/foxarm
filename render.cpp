#include <iostream>
#include <stdio.h>
#include <climits>

#include "GL/osmesa.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

extern "C"

float* render(double* projection,
              int im_height,
              int im_width,
              double* verts_buffer,
              int* tris_buffer,
              double* norms_buffer,
              int num_verts,
              int num_tris,
              int num_norms)
{
    printf("projection: %f %f %f\n", projection[0], projection[1], projection[2]);
    printf("im_height: %d\n", im_height);
    printf("im_width: %d\n", im_width);
    printf("verts_buffer: %f %f %f\n", verts_buffer[0], verts_buffer[1], verts_buffer[2]);
    printf("tris_buffer: %d %d %d\n", tris_buffer[0], tris_buffer[1], tris_buffer[2]);
    printf("norms_buffer: %f %f %f\n", norms_buffer[0], norms_buffer[1], norms_buffer[2]);
    printf("num_verts: %d\n", num_verts);
    printf("num_tris: %d\n", num_tris);
    printf("num_norms: %d\n", num_norms);

    float near = 0.05f;
    float far = 1e2f;
    float scale = (0x0001) << 0;

    OSMesaContext ctx;
    void *buffer;
    unsigned char* color_result = NULL;
    float* depth_result = NULL;

    double final_matrix[16];

    // create an RGBA-mode context
    ctx = OSMesaCreateContextExt(OSMESA_RGBA, 16, 0, 0, NULL );
    if (!ctx) {
        printf("OSMesaCreateContext failed!\n");
    }

    // allocate the image buffer
    buffer = malloc( im_width * im_height * 4 * sizeof(GLubyte) );
    if (!buffer) {
        printf("Alloc image buffer failed!\n");
    }

    // bind the buffer to the context and make it current
    if (!OSMesaMakeCurrent( ctx, buffer, GL_UNSIGNED_BYTE, im_width, im_height )) {
        printf("OSMesaMakeCurrent failed!\n");
    }
    OSMesaPixelStore(OSMESA_Y_UP, 0);

    // set color
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    // setup rendering
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // create projection
    double inv_width_scale  = 1.0 / (im_width * scale);
    double inv_height_scale = 1.0 / (im_height * scale);
    double inv_width_scale_1 = inv_width_scale - 1.0;
    double inv_height_scale_1_s = -(inv_height_scale - 1.0);
    double inv_width_scale_2 = inv_width_scale * 2.0;
    double inv_height_scale_2_s = -inv_height_scale * 2.0;
    double far_a_near = far + near;
    double far_s_near = far - near;
    double far_d_near = far_a_near / far_s_near;
    final_matrix[ 0] = projection[0+2*4] * inv_width_scale_1 + projection[0+0*4] * inv_width_scale_2;
    final_matrix[ 4] = projection[1+2*4] * inv_width_scale_1 + projection[1+0*4] * inv_width_scale_2;
    final_matrix[ 8] = projection[2+2*4] * inv_width_scale_1 + projection[2+0*4] * inv_width_scale_2;
    final_matrix[ 12] = projection[3+2*4] * inv_width_scale_1 + projection[3+0*4] * inv_width_scale_2;

    final_matrix[ 1] = projection[0+2*4] * inv_height_scale_1_s + projection[0+1*4] * inv_height_scale_2_s;
    final_matrix[ 5] = projection[1+2*4] * inv_height_scale_1_s + projection[1+1*4] * inv_height_scale_2_s;
    final_matrix[ 9] = projection[2+2*4] * inv_height_scale_1_s + projection[2+1*4] * inv_height_scale_2_s;
    final_matrix[13] = projection[3+2*4] * inv_height_scale_1_s + projection[3+1*4] * inv_height_scale_2_s;

    final_matrix[ 2] = projection[0+2*4] * far_d_near;
    final_matrix[ 6] = projection[1+2*4] * far_d_near;
    final_matrix[10] = projection[2+2*4] * far_d_near;
    final_matrix[14] = projection[3+2*4] * far_d_near - (2*far*near)/far_s_near;

    final_matrix[ 3] = projection[0+2*4];
    final_matrix[ 7] = projection[1+2*4];
    final_matrix[11] = projection[2+2*4];
    final_matrix[15] = projection[3+2*4];

    // load projection and modelview matrices
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(final_matrix);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /*
    unsigned char colorBytes[3];
    colorBytes[0] = (unsigned char)mat_props_buffer[0];
    colorBytes[1] = (unsigned char)mat_props_buffer[1];
    colorBytes[2] = (unsigned char)mat_props_buffer[2];
    */

    // render mesh
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, im_width, im_height);
    for (unsigned int i = 0; i < num_tris; ++i) {
        // glColor3ubv(colorBytes);
        glBegin(GL_POLYGON);

        unsigned int a = tris_buffer[3*i + 0];
        unsigned int b = tris_buffer[3*i + 1];
        unsigned int c = tris_buffer[3*i + 2];

        glNormal3dv(&norms_buffer[3 * a]);
        glVertex3dv(&verts_buffer[3 * a]);


        glNormal3dv(&norms_buffer[3 * b]);
        glVertex3dv(&verts_buffer[3 * b]);
        glNormal3dv(&norms_buffer[3 * c]);
        glVertex3dv(&verts_buffer[3 * c]);
        glEnd();
    }

    glFinish();

    // pull depth buffer and flip y axis
    GLint out_width, out_height, bytes_per_depth;
    GLboolean succeeded;
    unsigned short* p_depth_buffer;
    succeeded = OSMesaGetDepthBuffer(ctx, &out_width, &out_height, &bytes_per_depth, (void**)&p_depth_buffer);
    if (depth_result == NULL)
        depth_result = new float[out_width * out_height];
    for(int i = 0; i < out_width; i++){
        for(int j = 0; j < out_height; j++){
            int di = i + j * out_width; // index in depth buffer
            int ri = i + (out_height-1-j)*out_width; // index in rendered image
            if (p_depth_buffer[di] == USHRT_MAX) {
                depth_result[ri] = 0.0f;
            }
            else {
                depth_result[ri] = near / (1.0f - ((float)p_depth_buffer[di] / USHRT_MAX));
            }
        }
    }

    // free the image buffer
    free( buffer );

    // destroy the context
    OSMesaDestroyContext( ctx );
    return depth_result;
}
