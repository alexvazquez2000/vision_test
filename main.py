

#These need to be pip installed.
import cv2
import mediapipe as mp
import time

grid = True
result = None

# Function to draw a grid with coordinate labels on the frame
def draw_grid_with_labels(frame, num_rows=10, num_cols=10):
    h, w = frame.shape[:2]
    # Draw horizontal lines and y-coordinates
    for i in range(1, num_rows):
        y = h * i // num_rows
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(frame, str(y), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Draw vertical lines and x-coordinates
    for i in range(1, num_cols):
        x = w * i // num_cols
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Create a hand landmarker instance with the live stream mode:
def print_result(detection_result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global result
    if detection_result.hand_landmarks:
        result = detection_result
        print('hand landmarker result: {}'.format(result))

# Hand skeleton connections (mediapipe.solutions removed in 0.10 on Windows)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

#https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2)

with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    
    
    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
    
        # Flip the image horizontally for later selfie-view display
        frame = cv2.flip(frame, 1)
    
        # Draw grid with coordinate labels on the frame
        if cv2.waitKey(1) == ord('g'): grid = True
        if cv2.waitKey(1) == ord('h'): grid = False
        if grid: draw_grid_with_labels(frame)
    
        # Convert the BGR image to RGB
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


        # Returns time in nanoseconds as an integer
        frame_timestamp_ms =  time.monotonic_ns()
        
        # Send live image data to perform hand landmarks detection.
        # The results are accessible via the `result_callback` provided in
        # the `HandLandmarkerOptions` object.
        # The hand landmarker must be created with the live stream mode.
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Process the frame (result is updated by print_result callback)
        #hands.process(rgb_frame)
    
        # Draw hand landmarks and print thumb coordinates
        if result and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw all hand landmarks
                draw_hand_landmarks(frame, hand_landmarks)

                # Get thumb landmarks (landmark indices 1 to 4)
                thumb_landmarks = [hand_landmarks[i] for i in range(1, 5)]
    
                # Thumb tip is landmark 4 and thumb base is landmark 2
                thumb_tip = thumb_landmarks[3]
                thumb_base = thumb_landmarks[1]
    
                # Convert landmark position to pixel coordinates
                thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                thumb_base_coords = (int(thumb_base.x * frame.shape[1]), int(thumb_base.y * frame.shape[0]))

                #middle_tip = middle_landmarks[3]
                
                # Check if the thumb is up or down
                if thumb_tip_coords[1] < thumb_base_coords[1]:
                    print("Thumb is up!")
                    cv2.putText(frame, "Thumb Up!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Thumb is down!")
                    cv2.putText(frame, "Thumb Down!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        # Display the frame
        cv2.imshow('Hand Gesture Recognition with Grid and Coordinates', frame)
    
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
hand landmarker result: HandLandmarkerResult(handedness=[[Category(index=1, score=0.9798229932785034, display_name='Left', category_name='Left')]],
 hand_landmarks=[[
    NormalizedLandmark(x=0.42488721013069153, y=0.7947957515716553, z=4.212002977510565e-07, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.36050769686698914, y=0.7789446711540222, z=-0.02049502171576023, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.3035881519317627, y=0.7425565719604492, z=-0.033908579498529434, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.26493528485298157, y=0.713184118270874, z=-0.050584979355335236, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.24292683601379395, y=0.6809729933738708, z=-0.06576728075742722, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.33378368616104126, y=0.6018829941749573, z=-0.0009286391432397068, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.3282339572906494, y=0.530648946762085, z=-0.0264111440628767, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.3219883441925049, y=0.5079234838485718, z=-0.046078119426965714, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.31565427780151367, y=0.5039157867431641, z=-0.05658935382962227, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.3733232617378235, y=0.5811833739280701, z=-0.009827612899243832, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.38143086433410645, y=0.4783976674079895, z=-0.03443046286702156, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.36691829562187195, y=0.4261851906776428, z=-0.05401036888360977, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.35422927141189575, y=0.3927217125892639, z=-0.06490074098110199, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.4121968746185303, y=0.587461531162262, z=-0.02459540218114853, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.43753582239151, y=0.5073351860046387, z=-0.05839163437485695, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.4241234064102173, y=0.5421767234802246, z=-0.07167601585388184, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.4119972586631775, y=0.580674946308136, z=-0.0720851793885231, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.4521941840648651, y=0.6124377250671387, z=-0.04084845632314682, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.46986889839172363, y=0.5602073669433594, z=-0.06606670469045639, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.45292821526527405, y=0.579538881778717, z=-0.07162012159824371, visibility=None, presence=None, name=None),
     NormalizedLandmark(x=0.434733510017395, y=0.606266975402832, z=-0.07028892636299133, visibility=None, presence=None, name=None)
     ]],
     hand_world_landmarks=[[
        Landmark(x=0.02037077397108078, y=0.07848634570837021, z=0.03718109801411629, visibility=None, presence=None, name=None),
        Landmark(x=-0.009977729991078377, y=0.06351910531520844, z=0.022193901240825653, visibility=None, presence=None, name=None),
        Landmark(x=-0.03214344382286072, y=0.05211576819419861, z=0.013638527132570744, visibility=None, presence=None, name=None),
        Landmark(x=-0.05442973971366882, y=0.04312659054994583, z=-0.0060376315377652645, visibility=None, presence=None, name=None),
        Landmark(x=-0.065733402967453, y=0.02723059058189392, z=-0.017251577228307724, visibility=None, presence=None, name=None),
        Landmark(x=-0.023482922464609146, y=0.0028026546351611614, z=0.007793968543410301, visibility=None, presence=None, name=None),
        Landmark(x=-0.027129456400871277, y=-0.019091077148914337, z=-0.0021594271529465914, visibility=None, presence=None, name=None),
        Landmark(x=-0.031172070652246475, y=-0.027979010716080666, z=-0.017777832224965096, visibility=None, presence=None, name=None),
        Landmark(x=-0.03433879837393761, y=-0.02411135472357273, z=-0.0468333400785923, visibility=None, presence=None, name=None),
        Landmark(x=-0.004925168585032225, y=-0.003739937674254179, z=0.005337364040315151, visibility=None, presence=None, name=None),
        Landmark(x=-0.0022790832445025444, y=-0.038140345364809036, z=-0.006526223383843899, visibility=None, presence=None, name=None),
        Landmark(x=-0.011881149373948574, y=-0.05390053987503052, z=-0.02603316493332386, visibility=None, presence=None, name=None),
        Landmark(x=-0.021465517580509186, y=-0.06817019730806351, z=-0.04052894562482834, visibility=None, presence=None, name=None),
        Landmark(x=0.016870323568582535, y=-0.004076851531863213, z=-0.004643802065402269, visibility=None, presence=None, name=None),
        Landmark(x=0.019509796053171158, y=-0.023933248594403267, z=-0.018028499558568, visibility=None, presence=None, name=None),
        Landmark(x=0.01241447776556015, y=-0.016128571704030037, z=-0.03373315930366516, visibility=None, presence=None, name=None),
        Landmark(x=0.0025933887809515, y=0.00015129911480471492, z=-0.04004558175802231, visibility=None, presence=None, name=None),
        Landmark(x=0.03215637058019638, y=0.010495176538825035, z=-0.01149744726717472, visibility=None, presence=None, name=None),
        Landmark(x=0.03517455607652664, y=-0.0037836823612451553, z=-0.020643679425120354, visibility=None, presence=None, name=None), 
        Landmark(x=0.027626972645521164, y=-0.0014219465665519238, z=-0.03226142004132271, visibility=None, presence=None, name=None),
        Landmark(x=0.016065064817667007, y=0.009525304660201073, z=-0.0402543731033802, visibility=None, presence=None, name=None)
      ]])
'''
