#adapted from https://imagej.net/plugins/trackmate/scripting

import sys
import os

from ij import IJ
from ij import WindowManager

from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Settings
#from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import MaskDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer

direct = 'E:/Aaron/CK666_37C/Tracking_Images/'

#need windowless bioformats opening

for fi in os.listdir(direct):
	#fi = os.listdir(direct)[0]
	imm = [x for x in os.listdir(direct + fi) if '.tif' in x][0]
	path = direct+fi+'/'+imm
	imp = IJ.openImage(path)
	imp.show()
#	IJ.run("Re-order Hyperstack ...", "channels=[Frames (t)] slices=[Slices (z)] frames=[Channels (c)]")
	
	# We have to do the following to avoid errors with UTF8 chars generated in 
	# TrackMate that will mess with our Fiji Jython.
	reload(sys)
	sys.setdefaultencoding('utf-8')
	
	
	# Get currently selected image
	imp = WindowManager.getCurrentImage()
#	imp.show()
	
	
	#-------------------------
	# Instantiate model object
	#-------------------------
	
	model = Model()
	
	# Set logger
	#model.setLogger(Logger.IJ_LOGGER)
	
	#------------------------
	# Prepare settings object
	#------------------------
	
	settings = Settings(imp)
	
	# Configure detector
	settings.detectorFactory = MaskDetectorFactory()
	settings.detectorSettings = {
	    'SIMPLIFY_CONTOURS' : True,
	    'TARGET_CHANNEL' : 1
	}
	
	# Configure tracker
	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 10.0
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 6.0
	settings.trackerSettings['MAX_FRAME_GAP'] = 3
	
	# Add the analyzers for some spot features.
	# Here we decide brutally to add all of them.
	settings.addAllAnalyzers()
	
	# We configure the initial filtering to discard spots 
	# with a quality lower than 1.
	settings.initialSpotFilterValue = 1.
	
	#print(str(settings))
	
	#----------------------
	# Instantiate trackmate
	#----------------------
	
	trackmate = TrackMate(model, settings)
	
	#------------
	# Execute all
	#------------
	
	
	ok = trackmate.checkInput()
	if not ok:
	    sys.exit(str(trackmate.getErrorMessage()))
	
	ok = trackmate.process()
	if not ok:
	    sys.exit(str(trackmate.getErrorMessage()))
	
	
	
	#----------------
	# Display results
	#----------------
	
	#model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
	
	# A selection.
	sm = SelectionModel( model )
	
	# Read the default display settings.
	ds = DisplaySettingsIO.readUserDefault()
	
	# The viewer.
	displayer =  HyperStackDisplayer( model, sm, imp, ds ) 
	displayer.render()
	
	# The feature model, that stores edge and track features.
	fm = model.getFeatureModel()
	
	names = fm.getSpotFeatureNames()
	#model.getLogger().log('TRACK_ID, SPOT_ID, FRAME, POSITION_X, POSITION_Y, POSITION_Z')
	IJ.log("\\Clear")
	IJ.log('TRACK_ID,SPOT_ID,FRAME,POSITION_X,POSITION_Y,POSITION_Z')
	# Iterate over all the tracks that are visible.
	for id in model.getTrackModel().trackIDs(True):
	
	    # Fetch the track feature from the feature model.
	    #v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
	    #model.getLogger().log('')
	    #model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
	
		# Get all the spots of the current track.
	    track = model.getTrackModel().trackSpots(id)
	    for spot in track:
	        sid = spot.ID()
	        # Fetch spot features directly from spot.
	        # Note that for spots the feature values are not stored in the FeatureModel
	        # object, but in the Spot object directly. This is an exception; for tracks
	        # and edges, you have to query the feature model.
	        x=spot.getFeature('POSITION_X')*2
	        y=spot.getFeature('POSITION_Y')*2
	        z=spot.getFeature('POSITION_Z')*2
	        t=spot.getFeature('FRAME')
	        IJ.log(str(id) + ',' + str(sid)+ ',' + str(t) + ',' + str(x) + ',' + str(y) + ','+ str(z))
	

	filename = path.replace('_segmented.ome.tiff', '') + '_TrackMateLog.csv'
	if os.path.exists(filename):
		os.remove(filename)
	IJ.selectWindow("Log")
	IJ.saveAs("Text", filename)
	imp.close()
	IJ.log("\\Clear")
