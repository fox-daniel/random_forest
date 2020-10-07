import pandas as pd
import numpy as np

np.random.seed(23)


def make_one_d_points(xvals, classes):
    OneDPoints = pd.DataFrame(columns=["Feature_1", "Feature_2", "Target"])

    # features
    OneDPoints["Feature_1"] = xvals
    OneDPoints["Feature_2"] = np.full(OneDPoints["Feature_1"].shape[0], 0.5)
    # targets
    OneDPoints["Target"] = classes

    return OneDPoints


def make_four_corners():
    """Creates test data with a point in each quadrant.
    3 zeros, 1 one. No variables for calling the function."""
    TestFourCorners = pd.DataFrame(columns=["Feature_1", "Feature_2", "Target"])

    # features
    TestFourCorners["Feature_1"] = [0.25, 0.25, 0.75, 0.75]
    TestFourCorners["Feature_2"] = [0.25, 0.75, 0.25, 0.75]
    # targets
    TestFourCorners["Target"] = [0, 0, 0, 1]

    return TestFourCorners


def make_four_quadrants(num_points, classes, xbound, ybound):

    sides = [("<", "<"), ("<", ">"), (">", "<"), (">", ">")]
    xvalues = np.array([])
    yvalues = np.array([])
    targets = np.array([])

    for i, num in enumerate(num_points):

        xvals = np.random.random(num)
        yvals = np.random.random(num)
        outs = np.full(num, classes[i])
        side = sides[i]

        if side[0] == "<":
            xvals = xvals * xbound
        else:
            xvals = xbound + xvals * (1 - xbound)
        if side[1] == "<":
            yvals = yvals * ybound
        else:
            yvals = ybound + yvals * (1 - ybound)

        xvalues = np.concatenate([xvalues, xvals])
        yvalues = np.concatenate([yvalues, yvals])
        targets = np.concatenate([targets, outs])

    FourQuadrants = pd.DataFrame(
        {"Feature_1": xvalues, "Feature_2": yvalues, "Target": targets}
    )

    return FourQuadrants


def make_diagonal(num_points):
    """Makes random arrangement of points in [0,1]^2 with y=x line as boundary between classes

    Input: num_points
    Output: dataframe with points and classification
    """
    xvals = np.random.random(num_points)
    yvals = np.random.random(num_points)

    def rule(row):
        out = row.Feature_2 - row.Feature_1
        if out > 0:
            return 1
        else:
            return 0

    diagonal = pd.DataFrame({"Feature_1": xvals, "Feature_2": yvals})
    diagonal["Target"] = diagonal.apply(rule, axis=1)
    return diagonal


def make_diagonal_3d(num_points):
    xvals = np.random.random(num_points)
    yvals = np.random.random(num_points)
    zvals = np.random.random(num_points)

    def rule(row):
        out = row.Feature_3 - row.Feature_2 - row.Feature_1
        if out > 0:
            return 1
        else:
            return 0

    diagonal = pd.DataFrame(
        {"Feature_1": xvals, "Feature_2": yvals, "Feature_3": zvals}
    )
    diagonal["Target"] = diagonal.apply(rule, axis=1)
    return diagonal


def make_circle(n_points=700, frac_critical=0.3, overlap=0.25):

    num_min = int(n_points * frac_critical)
    num_maj = n_points - num_min
    r = 0.3
    center = 0.5
    # circle - minority
    noisiness = 0.09
    angle = np.random.random(num_min)
    noise_x = noisiness * (2 * np.random.random(num_min) - 1)
    noise_y = noisiness * (2 * np.random.random(num_min) - 1)
    x = (r + noise_x) * np.cos(2 * np.pi * angle) + center
    y = (r + noise_y) * np.sin(2 * np.pi * angle) + center
    minority = pd.DataFrame({"Feature_1": x, "Feature_2": y})

    # background - majority

    majority = pd.DataFrame(
        {"Feature_1": np.random.random(num_maj), "Feature_2": np.random.random(num_maj)}
    )

    def label_minority(row):
        return 1

    def label_majority(row):
        return 0

    minority["Target"] = minority.apply(label_minority, axis=1)
    majority["Target"] = majority.apply(label_majority, axis=1)

    def mark_overlap(row):
        u = row.Feature_1
        v = row.Feature_2
        d = np.sqrt((u - center) ** 2 + (v - center) ** 2)
        if d > r + overlap * noisiness:
            return True
        elif d < r - overlap * noisiness:
            return True
        else:
            return False

    def mask_overlap():
        """Removes the majority points that overlap with the minority."""
        mask = majority.apply(mark_overlap, axis=1)
        return mask

    mask = mask_overlap()
    majority = majority[mask]

    circle = pd.concat([minority, majority])
    circle = circle.sample(frac=1)
    circle.reset_index(inplace=True, drop=True)
    return circle


def make_diagonal_band(n_points=100, frac_minor=0.3, overlap=0.25):
    """Makes random arrangement of points in [0,1]^2 with y=x line as boundary between classes

    Input: num_points
    Output: dataframe with points and classification
    """

    num_min = int(n_points * frac_minor)
    num_maj = n_points - num_min

    # majority
    x = np.random.random(num_maj)
    y = np.random.random(num_maj)

    majority = pd.DataFrame({"Feature_1": x, "Feature_2": y})
    majority["Target"] = 0

    # minority
    x = np.random.random(num_min)
    noisiness = 0.09
    noise = noisiness * (2 * np.random.random(num_min) - 1)
    y = x + noise

    minority = pd.DataFrame({"Feature_1": x, "Feature_2": y})
    minority["Target"] = 1

    def mark_overlap(row):
        u = row.Feature_1
        v = row.Feature_2
        d = np.abs(v - u)
        if d > overlap * noisiness:
            return True
        else:
            return False

    def mask_overlap():
        """Removes the majority points that overlap with the minority."""
        mask = majority.apply(mark_overlap, axis=1)
        return mask

    mask = mask_overlap()
    majority = majority[mask]

    strip = pd.concat([minority, majority])
    strip = strip.sample(frac=1)
    strip.reset_index(inplace=True, drop=True)

    return strip


def make_octavo(
    num_points=np.full(8, 10),
    classes=[0, 0, 0, 0, 0, 0, 0, 1],
    xbound=0.5,
    ybound=0.5,
    zbound=0.5,
):
    """
    Creates data in 3D with each octavo of the unit cube all or the specified class
    Input:
    num_points: list of the number of points in each octavo
    classes: the class of the points in each octavo
    xbound, ybound, zbound: the bounds separating the octavos
    Output:
    dataframe with x,y,z coordinates and target classes
    """
    sides = [
        ("<", "<", "<"),
        ("<", "<", ">"),
        ("<", ">", "<"),
        ("<", ">", ">"),
        (">", "<", "<"),
        (">", "<", ">"),
        (">", ">", "<"),
        (">", ">", ">"),
    ]
    xvalues = np.array([])
    yvalues = np.array([])
    zvalues = np.array([])
    targets = np.array([])

    for i, num in enumerate(num_points):

        xvals = np.random.random(num)
        yvals = np.random.random(num)
        zvals = np.random.random(num)
        outs = np.full(num, classes[i])
        side = sides[i]

        if side[0] == "<":
            xvals = xvals * xbound
        else:
            xvals = xbound + xvals * (1 - xbound)
        if side[1] == "<":
            yvals = yvals * ybound
        else:
            yvals = ybound + yvals * (1 - ybound)
        if side[2] == "<":
            zvals = zvals * zbound
        else:
            zvals = zbound + zvals * (1 - zbound)

        xvalues = np.concatenate([xvalues, xvals])
        yvalues = np.concatenate([yvalues, yvals])
        zvalues = np.concatenate([zvalues, zvals])
        targets = np.concatenate([targets, outs])

    octavos = pd.DataFrame(
        {
            "Feature_1": xvalues,
            "Feature_2": yvalues,
            "Feature_3": zvalues,
            "Target": targets,
        }
    )

    return octavos


def make_diagonal_ndim(num_points, dim):
    X = np.random.random((num_points, dim))
    y = (X.sum(axis=1) > dim / 2).astype(int)
    data = pd.DataFrame(np.concatenate([X, y[:, np.newaxis]], axis=1))
    return data
