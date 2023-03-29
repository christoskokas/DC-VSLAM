#include "Frame.h"


namespace DC_VSLAM
{

ViewFrame::ViewFrame()
{
    
}

void ViewFrame::pangoQuit(Zed_Camera* zedPtr, const Map* _map)
{
    using namespace std::literals::chrono_literals;

    const int UI_WIDTH = 180;
    
    pangolin::CreateWindowAndBind("VSLAM", 1024,768);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768,500.0,500.0,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-1,-2, 0,0,0, pangolin::AxisNegY)
    );

    pangolin::Renderable renders;
    auto camera = std::make_shared<CameraFrame>();
    camera->zedCamera = zedPtr;
    camera->map = _map;
    
    // Create Interactive View in window
    pangolin::SceneHandler handler(renders, s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    d_cam.SetDrawFunction([&](pangolin::View& view){
        view.Activate(s_cam);
    });
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    camera->getOpenGLMatrix(Ow);
    camera->drawCamera();
    while( 1 )
    {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        camera->getOpenGLMatrix(Ow);
        camera->drawCamera();
        camera->drawKeyFrames();
        camera->lineFromKeyFrameToCamera();
        camera->drawPoints();
        s_cam.Follow(Ow);


        // Swap frames and Process Events
        pangolin::FinishFrame();

        if ( stopRequested )
        {
            pangolin::DestroyWindow("VSLAM");
            break;
        }

    }
    std::cout << "Visual Thread Exited!" << std::endl;
}

void ViewFrame::pangoQuitMulti(Zed_Camera* zedPtr, Zed_Camera* zedPtrB, const Map* _map)
{
    using namespace std::literals::chrono_literals;

    const int UI_WIDTH = 180;
    
    pangolin::CreateWindowAndBind("VSLAM", 1024,768);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024,768,500.0,500.0,512,389,0.1,1000),
        pangolin::ModelViewLookAt(0,-1,-2, 0,0,0, pangolin::AxisNegY)
    );

    pangolin::Renderable renders;
    auto camera = std::make_shared<CameraFrame>();
    camera->zedCamera = zedPtr;
    camera->zedCameraB = zedPtrB;
    camera->map = _map;
    
    // Create Interactive View in window
    pangolin::SceneHandler handler(renders, s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    d_cam.SetDrawFunction([&](pangolin::View& view){
        view.Activate(s_cam);
    });
    pangolin::OpenGlMatrix Twc, Twr;
    Twc.SetIdentity();
    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    camera->getOpenGLMatrix(Ow);
    camera->drawCamera();
    while( 1 )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        camera->getOpenGLMatrix(Ow);
        camera->drawCamera();
        camera->drawBackCamera();
        camera->drawKeyFrames();
        camera->lineFromKeyFrameToCamera();
        camera->drawPoints();
        s_cam.Follow(Ow);


        pangolin::FinishFrame();

        if ( stopRequested )
        {
            pangolin::DestroyWindow("VSLAM");
            break;
        }
    }
    std::cout << "Visual Thread Exited!" << std::endl;
}

void CameraFrame::drawPoints()
{
    glColor3f(1.0f,1.0f,1.0f);
    glBegin(GL_POINTS);


    std::unordered_map<unsigned long, MapPoint*> mapMapP = map->mapPoints;
    std::unordered_map<unsigned long, MapPoint*>::const_iterator itw, endw(mapMapP.end());
    for ( itw = mapMapP.begin(); itw != endw; itw ++)
    {
        if ( !(*itw).second )
            continue;
        if ( (*itw).second->GetIsOutlier() )
            continue;

        if ( (*itw).second->getActive() )
            glColor3f(0.0f,1.0f,0.0f);
        else
            glColor3f(1.0f,1.0f,1.0f);
        glVertex3d((*itw).second->wp3d(0),(*itw).second->wp3d(1),(*itw).second->wp3d(2));
    }
    glEnd();

}

void CameraFrame::drawKeyFrames()
{
    const int lastKeyFrameIdx {(int)map->kIdx - 1};
    if (lastKeyFrameIdx < 0)
        return;

    glColor3f(0.0f,1.0f,0.0f);
    std::unordered_map<unsigned long, KeyFrame*> mapKeyF = map->keyFrames;
    std::unordered_map<unsigned long,KeyFrame*>::const_iterator it, end(mapKeyF.end());
    for ( it = mapKeyF.begin(); it != end; it ++)
    {
        glColor3f(0.0f,0.0f,1.0f);

        if (!(*it).second->visualize)
            continue;
        glPushMatrix();
        Eigen::Matrix4d keyPose = (*it).second->getPose();
        glMultMatrixd((GLdouble*)keyPose.data());
        drawCameraFrame();
        glPopMatrix();

    }
}

void CameraFrame::drawCameraFrame()
{
    const float w = cameraWidth;
    const float h = w*0.75;
    const float z = w*0.3;
    glLineWidth(1);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();
}

void CameraFrame::getOpenGLMatrix(pangolin::OpenGlMatrix &MOw)
{
    for (int i = 0; i<4; i++) {
        MOw.m[4*i] = zedCamera->cameraPose.pose(4*i);
        MOw.m[4*i+1] = zedCamera->cameraPose.pose(4*i + 1);
        MOw.m[4*i+2] = zedCamera->cameraPose.pose(4*i + 2);
        MOw.m[4*i+3] = zedCamera->cameraPose.pose(4*i + 3);
    }
}

void CameraFrame::drawCamera()
{

    const float w = 2.0*cameraWidth;
    const float h = w*0.75;
    const float z = w*0.3;

    glPushMatrix();
    glColor3f(1.0f,1.0f,0.0f);


    glMultMatrixd((GLdouble*)zedCamera->cameraPose.pose.data());
    glLineWidth(2);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);

    glEnd();


    glPopMatrix();
}

void CameraFrame::drawBackCamera()
{

    const float w = 2.0*cameraWidth;
    const float h = w*0.75;
    const float z = w*0.3;
    const float off = +0.12;

    glPushMatrix();
    glColor3f(1.0f,1.0f,0.0f);

    glMultMatrixd((GLdouble*)zedCameraB->cameraPose.pose.data());
    glLineWidth(2);
    glBegin(GL_LINES);
    glVertex3f(off+0,0,0);
    glVertex3f(off+w,h,z);
    glVertex3f(off+0,0,0);
    glVertex3f(off+w,-h,z);
    glVertex3f(off+0,0,0);
    glVertex3f(off+-w,-h,z);
    glVertex3f(off+0,0,0);
    glVertex3f(off+-w,h,z);

    glVertex3f(off+w,h,z);
    glVertex3f(off+w,-h,z);

    glVertex3f(off+-w,h,z);
    glVertex3f(off+-w,-h,z);

    glVertex3f(off+-w,h,z);
    glVertex3f(off+w,h,z);

    glVertex3f(off+-w,-h,z);
    glVertex3f(off+w,-h,z);
    
    glEnd();

    glPopMatrix();
}

void CameraFrame::lineFromKeyFrameToCamera()
{
    const int lastKeyFrameIdx {(int)map->kIdx - 1};
    if (lastKeyFrameIdx < 0)
        return;
    glPushMatrix();
    glLineWidth(1);
    std::unordered_map<unsigned long, KeyFrame*> mapKeyF = map->keyFrames;
    std::unordered_map<unsigned long, KeyFrame*>::const_iterator it, end(mapKeyF.end());
    glBegin(GL_LINES);
    for ( it = mapKeyF.begin(); it != end; it ++)
    {
        const KeyFrame* kfCand = it->second;
        if (!kfCand->visualize)
            continue;
        glColor3f(1.0f,0.0f,0.0f);
        if ( !kfCand->nextKF )
            continue;
        const KeyFrame* nextKF = kfCand->nextKF;
        glVertex3f((GLfloat)nextKF->pose.pose(0,3),(GLfloat)nextKF->pose.pose(1,3),(GLfloat)nextKF->pose.pose(2,3));
        glVertex3f((GLfloat)kfCand->pose.pose(0,3), (GLfloat)kfCand->pose.pose(1,3), (GLfloat)kfCand->pose.pose(2,3));
    }
    glEnd();
    glColor3f(0.0f,1.0f,0.0f);
    Eigen::Matrix4d keyPose = mapKeyF.at(lastKeyFrameIdx)->getPose();
    Eigen::Matrix4d camPose = zedCamera->cameraPose.pose;
    glBegin(GL_LINES);
    glVertex3f((GLfloat)keyPose(0,3),(GLfloat)keyPose(1,3),(GLfloat)keyPose(2,3));
    glVertex3f((GLfloat)camPose(0,3), (GLfloat)camPose(1,3), (GLfloat)camPose(2,3));
    glEnd();
    glPopMatrix();
}

}