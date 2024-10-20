# return overlapping_corners, area
def check_overlap_for_2(box1, box2):
    # bbox is TL and BR corner coordinates
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    box1_corners = [
        (x1, y1),  # TL
        (x2, y1),  # TR
        (x1, y2),  # BL
        (x2, y2),  # BR
    ]

    overlapping_corners = False
    for corner in box1_corners:
        x, y = corner
        # check if x1,y1 is in box2
        if (x >= a1 and x <= a2) and (y >= b1 and y <= b2):
            overlapping_corners = True
            break

    if not overlapping_corners:
        for corner in box1_corners:
            a, b = corner
            # check if x1,y1 is in box2
            if (a >= x1 and a <= x2) and (b >= y1 and b <= y2):
                overlapping_corners = True
                break

    area = 0
    if overlapping_corners:
        # calculate the coordinates of the overlapping rectangle
        x_left = max(x1, a1)
        y_top = max(y1, b1)
        x_right = min(x2, a2)
        y_bottom = min(y2, b2)

        # calculate overlap area
        overlap_width = x_right - x_left
        overlap_height = y_bottom - y_top
        area = overlap_width * overlap_height

    return area
