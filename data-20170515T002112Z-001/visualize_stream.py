import argparse, cv2, os, simplejson, sys

if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser(description='Stream camera data.')
    parser.add_argument("--input", help="input video file", required=True)
    args = parser.parse_args()

    # Load stream, layout, and events.
    cap = cv2.VideoCapture(args.input)
    layout = simplejson.loads(open(args.input.replace(".avi", "-layout.json")).read())
    events = simplejson.loads(open(args.input.replace(".avi", "-events.json")).read())
    
    frame  = 0
    ret, im = cap.read()
    
    print("Press 'q' to quit. Any other key to advance.")
    while ret:
        # Display.
        vis_im = 1*im
        for (food_name, poly) in layout.items():
            poly = map(tuple, poly)
            c = (0,0,0)
            for event in events:
                if food_name==event["object-class"] and \
                   frame >= event["start-frame"] and frame <= event["end-frame"]:
                    c = (0,255,0) if "added" in event["event-class"] else (0,0,255)
            [cv2.circle(vis_im, v, 3, c, 1) for v in poly]
            [cv2.line(vis_im, u, v, c, 1) for (u,v) in zip(poly, poly[1:]+[poly[0]])]
        cv2.imshow(__file__, vis_im)
        c = cv2.waitKey(1)
        c = None if c not in range(256) else chr(c)
        
        # Keypress handling
        if c == 'q':
            break
        elif c is not None:
            ret, im = cap.read()
            frame += 1
            for event in events:
                if event["start-frame"] == frame:
                    print("Start %s %s" % (event["object-class"], event["event-class"]))
                if event["end-frame"] == frame:
                    print("End %s %s" % (event["object-class"], event["event-class"]))
