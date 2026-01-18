# state file generated using paraview version 6.0.0-RC1
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[793, 816],
    CenterOfRotation=[2540.0, -440.0, 150.0],
    CameraPosition=[24.008628973028205, 2308.8371595153526, 1184.1399391441448],
    CameraFocalPoint=[2539.9999999999995, -439.99999999999994, 150.0],
    CameraViewUp=[0.2911718565454076, -0.09174275996358917, 0.952261632089713],
    CameraFocalDisk=1.0,
    CameraParallelScale=3808.097628363439,
    OSPRayMaterialLibrary=materialLibrary1,
)

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(793, 816)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML MultiBlock Data Reader'
surfacesvtm = XMLMultiBlockDataReader(registrationName='surfaces.vtm', FileName=['/home/jb/workspace/HLAVO/deep_model/gpt_zone/src/model/surfaces.vtm'])
surfacesvtm.TimeArray = 'None'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from surfacesvtm
surfacesvtmDisplay = Show(surfacesvtm, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
vtkBlockColorsLUT.Set(
    InterpretValuesAsCategories=1,
    AnnotationsInitialized=1,
    Annotations=['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11'],
    ActiveAnnotatedValues=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    IndexedColors=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.63, 0.63, 1.0, 0.67, 0.5, 0.33, 1.0, 0.5, 0.75, 0.53, 0.35, 0.7, 1.0, 0.75, 0.5],
)

# trace defaults for the display properties.
surfacesvtmDisplay.Set(
    Representation='Surface',
    ColorArrayName=['FIELD', 'vtkBlockColors'],
    LookupTable=vtkBlockColorsLUT,
    Assembly='Hierarchy',
    BlockSelectors=['/Root/layer_1_relief', '/Root/layer_2_Q1_top', '/Root/layer_3_Q1_base', '/Root/layer_4_Q2_top', '/Root/layer_5_Q2_base', '/Root/layer_6_Q3_top', '/Root/layer_7_Q3_base', '/Root/layer_8_Q5_top', '/Root/layer_9_Q5_base', '/Root/layer_10_Q6_top', '/Root/layer_11_Q6_base', '/Root/layer_12_TNgmerged', '/Root/layer_13_TCb3merged', '/Root/layer_14_TNwmerged', '/Root/layer_15_TCb2merged', '/Root/layer_16_TMwmerged', '/Root/layer_17_TCb1merged', '/Root/layer_18_Pwmerged'],
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
surfacesvtmDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 18.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
surfacesvtmDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 18.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for vtkBlockColorsLUT in view renderView1
vtkBlockColorsLUTColorBar = GetScalarBar(vtkBlockColorsLUT, renderView1)
vtkBlockColorsLUTColorBar.Set(
    Title='vtkBlockColors',
    ComponentTitle='',
)

# set color bar visibility
vtkBlockColorsLUTColorBar.Visibility = 1

# show color legend
surfacesvtmDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'vtkBlockColors'
vtkBlockColorsPWF = GetOpacityTransferFunction('vtkBlockColors')

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(surfacesvtm)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------