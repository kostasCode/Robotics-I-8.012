import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter

# PART A

def plot_free_vec(x,c="b"):
    """Κατασκευάστε συνάρτηση (function) στην Python, η οποία
    να εμφανίζει σε figure ένα διάνυσμα x, στον 2-διάστατο ή
    στον 3-διάστατο χώρο (θα πρέπει αυτό να αναγνωρίζεται
    αυτόματα), το οποίο θα βρίσκεται στην αρχή των αξόνων
    {0}. Το διάνυσμα να εμφανίζεται ως συνεχής γραμμή
    πάχους 2 με άκρο «ο». Η αρχή των αξόνων να απεικονίζεται
    ως «*». Στην συνάρτηση θα πρέπει ο χρήστης να μπορεί να
    επιλέξει το χρώμα του διανύσματος μέσω του ορίσματος c.
    Επαληθεύστε την συνάρτηση δίνοντας τυχαία διανύσματα,
    τόσο στον 2-διάστατο χώρο, όσο και στον 3-διάστατο. """

    plt.figure()
    if len(x) == 2:
        ax = plt.axes()
        ax.plot([0,x[0]],[0,x[1]],c+"-",linewidth=2)
        ax.plot(x[0],x[1],c+"o")
        ax.plot(0,0,c+"*")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2d plot')

    elif len(x) == 3:
        ax = plt.axes(projection='3d')
        ax.plot([0,x[0]],[0,x[1]],[0,x[2]],c+"-",linewidth=2)
        ax.plot(x[0],x[1],x[2],c+"o")
        ax.plot(0,0,0,c+"*")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3d plot')
    else:
        raise ValueError("Invalid dimension")
    ax.set_aspect('equal')

    plt.axis('equal')
    plt.draw()
    plt.show()

def plot_vec(x, s, c="b", axis=None): # axis is not corelated in viewing axis ratio
    """
    Κατασκευάστε παρόμοια συνάρτηση με την άσκηση 1, η
    οποία αυτή την φορά θα εκτυπώνει το διάνυσμα x
    τοποθετημένο σε κάποιο σημείο που θα δίνεται από το
    όρισμα s. Επαληθεύστε την συνάρτηση δίνοντας τυχαία
    διανύσματα, τόσο στον 2-διάστατο χώρο, όσο και στον 3-διάστατο.
    """
    
    if len(x) == 2:
        if axis is None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        ax.plot([s[0], x[0]], [s[1], x[1]], c + "-", linewidth=2)
        ax.plot(x[0], x[1], c + "o")
        ax.plot(s[0], s[1], c + "*")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D Vector')

    elif len(x) == 3:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axis

        ax.plot([s[0], x[0]], [s[1], x[1]], [s[2], x[2]], c + "-", linewidth=2)
        ax.plot([x[0]], [x[1]], [x[2]], c + "o")
        ax.plot([s[0]], [s[1]], [s[2]], c + "*")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('3D Vector')

    else:
        raise ValueError("Input vectors must be 2D or 3D.")

    plt.axis('equal')
    plt.draw()

    return ax

def make_unit(x):
    """Κατασκευάστε συνάρτηση, η οποία θα βρίσκει το
    μοναδιαίο διάνυσμα κατεύθυνσης ενός διανύσματος χ
    (πιθανώς μη-μοναδιαίου). Επαληθεύστε την συνάρτηση
    δίνοντας τυχαία διανύσματα, τόσο στον 2-διάστατο χώρο, όσο
    και στον 3-διάστατο."""

    norm = np.linalg.norm(x) # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    if norm == 0:
        raise ValueError("Το διάνυσμα μηδέν δεν έχει μοναδιαίο διάνυσμα")
    return x / norm

def project_vec(a,b):
    """Κατασκευάστε συνάρτηση, η οποία θα προβάλει το 
    διάνυσμα b στο διάνυσμα a. Αν το διάνυσμα a, δεν είναι 
    μοναδιαίο, θα πρέπει να χρησιμοποιήσετε το μοναδιαίο 
    διάνυσμα της κατεύθυνσής του. Επαληθεύστε την 
    συνάρτηση δίνοντας τυχαία διανύσματα, τόσο στον 2-
    διάστατο χώρο, όσο και στον 3-διάστατο.
    """
    
    n = make_unit(a).reshape(-1, 1)  
    b_col = b.reshape(-1, 1)         

    projection_matrix = (n @ n.T) / (n.T @ n)

    projected = (projection_matrix @ b_col).flatten()

    print("Προβολή:", projected)

    origin = np.zeros_like(a)
    ax = plot_vec(projected, origin, "r")
    plot_vec(b, origin, "g", ax)
    plot_vec(a, origin, "y", ax)

    plt.title("Προβολή του b στο a (μέσω πίνακα προβολής)")
    plt.show()

    return projected
    # explenation req at report 2d 3d long a or b + s

def cross_demo(s1, a, s2, b):
    """Κατασκευάστε script το οποίο να παρουσιάζει γραφικά την 
    πράξη τoυ εξωτερικού γινομένου για δύο τυχαία 
    διανύσματα."""
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    cross = np.cross(a-s1,b-s2)
    
    s1=make_unit(s1)
    a=make_unit(a)
    s2=make_unit(s2)
    b=make_unit(b)
    cross=make_unit(cross)
    
    plot_vec(s1,a,"r",ax)
    plot_vec(s2,b,"g",ax)
    plot_vec(cross,a,"b",ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Cross Product Demo')

    plt.show()

def plot_Rot(rot_m):
    """Κατασκευάστε συνάρτηση στην Python, που να δέχεται 
    πίνακα στροφής R και να εμφανίζει το πλαίσιο 
    συντεταγμένων γραφικά (σε figure). Επαληθεύστε για 
    κάποιους συγκεκριμένους πίνακες στροφής"""
    
    origin = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    cross = np.array([0,0,1])
    
    normalized_frame = np.array([v1,v2,cross])
    
    result = rot_m @ normalized_frame
    
    axis = plot_vec(result[0],origin,"r")
    plot_vec(result[1],origin,"g",axis)
    plot_vec(result[2],origin,"b",axis)
    
    print(cross)
    plt.show()

def generate_rot():
    """Κατασκευάστε συνάρτηση η οποία να δημιουργεί και να 
    παρουσιάζει τυχαίο πίνακα στροφής (R) στον 3-διάστατο 
    χώρο. Επαληθεύστε την συνάρτηση χρησιμοποιώντας την 
    plot_Rot."""
    
    # https://www.geeksforgeeks.org/random-numbers-in-python/
    roll = -np.radians(random.randrange(20, 50, 3))
    pitch = -np.radians(random.randrange(20, 50, 3))
    yaw = - np.radians(random.randrange(20, 50, 3))
    print(roll,pitch,yaw)
    return rotX(roll) @ rotY(pitch) @ rotZ(yaw)

# PART B

"""Κατασκευάστε συναρτήσεις στην Python, οι οποίες να 
δέχονται γωνία σε μοίρες th (σε rad) και να 
επιστρέφουν τον πίνακα στροφής R που αντιστοιχεί 
σε στροφή γύρω από τον άξονα Χ, Υ και Ζ αντίστοιχα. 
Επαληθεύστε γραφικά τις συναρτήσεις δίνοντας τυχαίες 
γωνίες και χρησιμοποιώντας την plot_Rot. 
"""

def rotX(th):
    """
    Rotation around X-axis (Roll)
    θ in radians
    """
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def rotY(th):
    """
    Rotation around Y-axis (Pitch)
    θ in radians
    """
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotZ(th):
    """
    Rotation around Z-axis (Yaw)
    θ in radians
    """
    c = np.cos(th)
    s = np.sin(th)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def plot_hom(G,scale=None,axis=None):
    """Κατασκευάστε συνάρτηση στην Python, που να 
    δέχεται ομογενή μετασχηματισμό G και να εμφανίζει 
    το πλαίσιο συντεταγμένων γραφικά (σε figure) 
    τοποθετημένο στον χώρο. Η μεταβλητή scale θα 
    καθορίζει το μήκος των διανυσμάτων κατά την 
    εμφάνιση σε μέτρα. Επαληθεύστε για κάποιους 
    συγκεκριμένους ομογενείς μετασχηματισμούς.
    """

    origin = np.array([0,0,0,1])
    v1 = np.array([1,0,0,1])
    v2 = np.array([0,1,0,1])
    v3 = np.array([0,0,1,1])
    
    origin = G @ origin
    v1 = G @ v1
    v2 = G @ v2
    v3 = G @ v3
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
    else:
        axis = axis
    
    
    if scale:
        v1[:3] = origin[:3] + scale * (v1[:3] - origin[:3])
        v2[:3] = origin[:3] + scale * (v2[:3] - origin[:3])
        v3[:3] = origin[:3] + scale * (v3[:3] - origin[:3])
    
    axis = plot_vec(v1[:3], origin[:3], "r",axis)
    axis = plot_vec(v2[:3], origin[:3], "g", axis)
    axis = plot_vec(v3[:3], origin[:3], "b", axis)
    
    plt.show()
    return axis

def homogen(R, p):
    return np.vstack((np.hstack((R, p.reshape(3,1))), [0,0,0,1]))

def gr(R): return homogen(R, np.zeros(3))

def gp(p): return homogen(np.eye(3), p)

def gRX(theta): return gr(rotX(theta))

def gRY(theta): return gr(rotY(theta))

def gRZ(theta): return gr(rotZ(theta))

def rotAndTranVec(G, vin):
    v = np.append(vin, 1)
    return (G @ v)[:3]

# helper function for rotAndTrans_shape
def data_for_cylinder_along_z(center_x, center_y, radius, height_z, resolution=50):
    """
    Δημιουργεί και επιστρέφει τα σημεία επιφάνειας κυλίνδρου με άξονα τον Ζ.
    
    Είσοδοι:
    - center_x, center_y: μετατόπιση του κυλίνδρου στην xy-επίπεδο
    - radius: ακτίνα του κυλίνδρου
    - height_z: ύψος του κυλίνδρου κατά τον άξονα z
    - resolution: αριθμός σημείων κατά γωνία και κατά ύψος
    
    Έξοδοι:
    - X, Y, Z: πίνακες σημείων (meshgrid), έτοιμοι για χρήση σε plot ή μετασχηματισμό
    """
    z = np.linspace(0, height_z, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid

def rotAndTrans_shape(X, Y, Z, G):
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten(), np.ones(X.size)))
    transformed = G @ points
    return transformed[0].reshape(X.shape), transformed[1].reshape(Y.shape), transformed[2].reshape(Z.shape)

# PART C

# helper function for g0e
def MDH(a, alpha, theta, d):
    """
    Modified Denavit-Hartenberg μετασχηματισμός:
    T = TransX(a) · RotX(alpha) · RotZ(theta) · TransZ(d)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st,  0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa,  ca,  ca * d],
        [0, 0, 0, 1]
    ])

def g0e(q1, q2, q3):
    """
    Επιστρέφει: 4x4 πίνακας G
    """
    # Σταθερές του ρομπότ (μήκη βραχιόνων)
    L1 = 1.0
    L2 = 1.0
    
    # Πίνακες MDH για κάθε άρθρωση:
    A1 = MDH(a=0,  alpha=0, theta=0,  d=q1)   # Πρισματική
    A2 = MDH(a=0, alpha=0, theta=q2, d=0)    # Περιστροφική
    A3 = MDH(a=0, alpha=np.radians(90), theta=q3, d=L1)    # Περιστροφική
    
    G = A1 @ A2 @ A3
    return G

def plot_robot(q1, q2, q3,axis = None, plot = True):
    """
    Σχεδιάζει το PRR ρομπότ ως σκελετό με διανύσματα που ενώνουν τις αρθρώσεις,
    χρησιμοποιώντας Modified DH μετασχηματισμούς κατα Craig.
    """
    global L1
    L1 = 1.0 
    L2 = 2.0  

    A1 = MDH(a=0, alpha=0, theta=0, d=q1)
    A2 = MDH(a=0, alpha=0, theta=q2, d=0)
    A3 = MDH(a=0, alpha=np.radians(90), theta=q3, d=L1)
    A4 = gp(np.array([0, -L2, 0]))

    G0 = np.eye(4)
    G1 = A1
    G2 = G1 @ A2
    G3 = G2 @ A3
    G4 = G3 @ A4

    p0 = G0[:3, 3]
    p1 = G1[:3, 3]
    p2 = G2[:3, 3]
    p3 = G3[:3, 3]
    p4 = G4[:3, 3]

    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
    
    if(plot):
        plot_vec(p1, p0, "r",ax)       # συνδεσμος 1
        plot_vec(p2, p1, "g", ax)       # συνδεσμος 2
        plot_vec(p3, p2, "b", ax)       # συνδεσμος 3
        plot_vec(p4, p3, "c", ax)       # end effector

        # labels
        ax.text(*p0, "Base,Joint 1", fontsize=10)
        ax.text(*p1, "Joint 2,Joint 3", fontsize=10)
        #ax.text(*p3, "Joint 3", fontsize=10)
        ax.text(*p4, "EE", fontsize=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("PRR Robot (Vector Visualization)")
        ax.set_box_aspect([1, 1, 1])
        plt.show()
            
    return p4

def workspace_demo(plot=True):
    """
    Υπολογίζει και εμφανίζει τον χώρο εργασίας του PRR ρομπότ
    βασισμένο ΜΟΝΟ στη θέση του end-effector (άκρο ρομπότ).
    
    plot_sample: αν True, σχεδιάζει δείγματα του ρομπότ.
    """
    q1_range = np.linspace(0.7, 3.0, 70) 
    q2_range = np.radians(np.linspace(-170, 170, 80))
    q3_range = np.radians(np.linspace(-135, 135, 80))

    data = []

    for i, q1 in enumerate(q1_range):
        for q2 in q2_range:
            for q3 in q3_range:
                A1 = MDH(a=0, alpha=0, theta=0, d=q1)
                A2 = MDH(a=0, alpha=0, theta=q2, d=0)
                A3 = MDH(a=0, alpha=np.radians(90), theta=q3, d=1.0)
                A4 = gp(np.array([0, -2, 0]))  

                G = A1 @ A2 @ A3 @ A4
                p = G[:3, 3]
                data.append([p[0], p[1], p[2], q1, q2, q3])

    data = np.array(data) # neccesary !!!
    
    if(plot):
        points = data[:, :3]     
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='navy')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Χώρος Εργασίας PRR Ρομπότ (End-effector μόνο)")
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    return data  # για επόμενη χρήση στο C3

def ikine(p_goal, workspace_points, joint_configs):
    """
    Λύνει το αντίστροφο κινηματικό πρόβλημα για το PRR ρομπότ.

    Είσοδος:
    - p_goal: επιθυμητό σημείο (3D numpy array)
    - workspace_points: πίνακας Nx3 με σημεία του χώρου εργασίας (από C2)
    - qs: πίνακας Nx3 με τις αντίστοιχες τιμές q1, q2, q3

    Έξοδος:
    - q1, q2, q3: άρθρωση που προσεγγίζει καλύτερα το p_goal
    - d: απόσταση (σφάλμα) από το p_goal
    """

    distances = np.linalg.norm(workspace_points - p_goal, axis=1)
    
    if len(distances) == 0 or np.min(distances) > 2:
        raise ValueError("Target point is unreachable or not in workspace.")

    idx_min = np.argmin(distances)

    q1, q2, q3 = joint_configs[idx_min]
    
    d = distances[idx_min]

    return q1, q2, q3, d

def get_trajectory(p0, pf, t, tf, order=5):
    """
        Κατασκευάστε συνάρτηση που να δέχεται την αρχική θέση 
        p0, την τελική θέση pf, τον τρέχοντα χρόνο t, και τον 
        συνολικό χρόνο tf και να υπολογίζει την θέση, την 
        ταχύτητα και την επιτάχυνση την τρέχουσα χρονική 
        στιγμή της τροχιάς που έχει σχεδιαστεί μέσω 
        πεμπτοβάθμιου πολυωνύμου με αρχική και τελική 
        επιτάχυνση και ταχύτητα ίση με μηδέν. Η συνάρτηση θα 
        πρέπει να λειτουργεί για p οποιασδήποτε διάστασης (π.χ. 
        τόσο για τον χώρο εργασίας, όσο και για τον χώρο των 
        αρθρώσεων). Επαληθεύστε την συνάρτηση δίνοντας 
        γραφήματα τόσο για την θέση, όσο και για την ταχύτητα. 
        Επαναλάβετε την ίδια διαδικασία για τροχιά τριτοβάθμιου 
        πολυωνύμου. Συγκρίνετε και σχολιάστε τα αποτελέσματα.
    """
    tau = t / tf

    if order == 3:
        tau2 = tau**2
        tau3 = tau**3
        u = p0 + (pf - p0) * (3*tau2 - 2*tau3)
        u_dot = (pf - p0) * (6*tau - 6*tau2) / tf
        u_ddot = (pf - p0) * (6 - 12*tau) / tf**2
    elif order == 5:
        tau2 = tau**2
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5
        u = p0 + (pf - p0) * (10*tau3 - 15*tau4 + 6*tau5)
        u_dot = (pf - p0) * (30*tau2 - 60*tau3 + 30*tau4) / tf
        u_ddot = (pf - p0) * (60*tau - 180*tau2 + 120*tau3) / tf**2
    else:
        raise ValueError("order 3 or 5")

    return u, u_dot, u_ddot

from scipy.signal import savgol_filter

def helper_plot(pA, pB, tf, points, joint_values, order=3, space="joint", plot=True):
    """
    Οπτικοποιεί την τροχιά σε joint ή workspace ανάλογα με το space.
    
    space = "joint" -> δίνεται pA, pB σε x,y,z και βρίσκονται οι αρθρώσεις q1 q2 q3
    space = "work"  -> δίνεται pA, pB σε q1,q2,q3 και υπολογίζεται η θέση μέσω plot_robot
    """
    t_vals = np.linspace(0, tf, 100)
    xyz_vals = []
    q_vals = []

    for t in t_vals:
        if space == "joint":
            p, _, _ = get_trajectory(np.array(pA), np.array(pB), t, tf, order=order)
            q1, q2, q3, _ = ikine(p, points, joint_values)
            xyz_vals.append(p)
            q_vals.append([q1, q2, q3])
        elif space == "work":
            q, _, _ = get_trajectory(np.array(pA), np.array(pB), t, tf, order=order)
            q1, q2, q3 = q
            q_vals.append([q1, q2, q3])
            
            A1 = MDH(a=0, alpha=0, theta=0, d=q1)
            A2 = MDH(a=0, alpha=0, theta=q2, d=0)
            A3 = MDH(a=0, alpha=np.radians(90), theta=q3, d=1)
            A4 = gp(np.array([0, -2, 0]))

            G1 = A1
            G2 = G1 @ A2
            G3 = G2 @ A3
            G4 = G3 @ A4
            p4 = G4[:3, 3]
            xyz_vals.append(p4)
        else:
            raise ValueError("Το όρισμα 'space' πρέπει να είναι 'joint' ή 'work'.")

    xyz_vals = np.array(xyz_vals)
    q_vals = np.array(q_vals)

    # Φιλτράρισμα αρθρώσεων
    window_length = 11
    polyorder = 3
    if len(q_vals) >= window_length:
        for i in range(3):
            q_vals[:, i] = savgol_filter(q_vals[:, i], window_length, polyorder)

    # Αν plot=False, επιστρέφει δεδομένα
    if not plot:
        return q_vals, xyz_vals

    # Παράγωγοι αρθρώσεων
    q_dot_vals = np.gradient(q_vals, axis=0) / (t_vals[1] - t_vals[0])
    q_ddot_vals = np.gradient(q_dot_vals, axis=0) / (t_vals[1] - t_vals[0])

    # Τροχιά στο χώρο
    colors = (t_vals - t_vals.min()) / (t_vals.max() - t_vals.min())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xyz_vals[:, 0], xyz_vals[:, 1], xyz_vals[:, 2],
                    c=colors, cmap='viridis', marker='o')
    ax.set_title("Τροχιά στον χώρο εργασίας (χρώμα ~ χρόνος)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    cbar = plt.colorbar(sc, ax=ax, label='Χρόνος [s]')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"{t_vals.min():.1f}", f"{t_vals.max():.1f}"])
    plt.show()

    # Θέσεις αρθρώσεων
    joint_labels = ['q1 (m)', 'q2 (rad)', 'q3 (rad)']
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        axs[i].plot(t_vals, q_vals[:, i], label="Θέση")
        axs[i].set_title(f"Αρθρωση {joint_labels[i]}")
        axs[i].set_xlabel("Χρόνος [s]")
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

    # Συντεταγμένες θέσης
    xyz_dot = np.gradient(xyz_vals, axis=0) / (t_vals[1] - t_vals[0])
    xyz_ddot = np.gradient(xyz_dot, axis=0) / (t_vals[1] - t_vals[0])
    xyz_labels = ['x', 'y', 'z']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        axs[i].plot(t_vals, xyz_vals[:, i], label="Θέση")
        axs[i].plot(t_vals, xyz_dot[:, i], label="Ταχύτητα", linestyle='--')
        axs[i].plot(t_vals, xyz_ddot[:, i], label="Επιτάχυνση", linestyle=':')
        axs[i].set_title(f"Συντεταγμένη {xyz_labels[i]}")
        axs[i].set_xlabel("Χρόνος [s]")
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #Α1
    # v1 = np.array([50,60])
    # s1 = np.array([5,6])

    # v2 = np.array([10,20,30])
    # s2 = np.array([1,2,30])

    # plot_free_vec(v1,'r')
    # plot_free_vec(v2,'r')


    #Α2
    # v1 = np.array([50,60])
    # s1 = np.array([5,6])
    # v2 = np.array([10,20,30])
    # s2 = np.array([1,2,30])
    # plot_vec(v1,s1,'r')
    # plot_vec(v2,s2,'r')
    # plt.show()


    #Α3 
    # v1 = np.array([50,60])
    # v2 = np.array([10,20,30])
    # print(make_unit(v1))
    # print(make_unit(v2))


    #A4 
    v1 = np.array([4,0])
    s1 = np.array([3,2])
    print(project_vec(v1,s1))
    v1 = np.array([10, 20, 5])
    v2 = np.array([5, 10, 0])  

    project_vec(v1,v2)
    
    
    #A5 - script     
    # origin = np.array([1, 1, 1])
    # v1 = np.array([20, 0, 0])
    # v2 = np.array([0, 20 , 0])
    # cross_demo(v1,origin,v2,origin)


    #A6
    # theta_deg = 90
    # theta_rad = -np.radians(theta_deg)
    # plot_Rot(rotX(theta_rad))


    #A7
    # plot_Rot(generate_rot())
    
    # PART B
    
    #B1 use of those function demonstrated by Bi where i belongs in PART B
    
    #B2
    # rot = rotZ(np.radians(-90))
    # move = np.array([0,0,0])       
    # row1 = np.append(rot[0],move[0])
    # row2 = np.append(rot[1],move[1])
    # row3 = np.append(rot[2],move[2])
    # standard = np.array([0,0,0,1])
    # # Stack all rows into a full 4x4 matrix
    # G = np.array([row1, row2, row3, standard])    
    # plot_hom(G,scale=0.5)
    
    #B3    
    # plot_hom(homogen(rotX(np.radians(45)),np.array([1,2,3])),scale=1.5)

    #B4
    # plot_hom(gr(rotZ(np.radians(45))))
    
    #B5
    # plot_hom(gp(np.array([1,1,1])))
    
    # B7
    # X, Y, Z = data_for_cylinder_along_z(center_x=0, center_y=0, radius=0.5, height_z=1.0)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, color='lightblue', edgecolor='k', alpha=0.7)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.title("Επιφάνεια Κυλίνδρου")
    # plt.show()
    # G = homogen(rotX(np.radians(90)), np.array([0.5, 0.5, 0]))
    # Xt, Yt, Zt = rotAndTrans_shape(X, Y, Z, G)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.plot_surface(Xt, Yt, Zt, color='lightgreen', edgecolor='k', alpha=0.7)
    # plt.title("Μετασχηματισμένη Επιφάνεια Κυλίνδρου")
    # plt.show()
    
    # PART C

    # C1
    # plot_robot(q1=1.0, q2=np.radians(0), q3=np.radians(0))
    # plot_robot(q1=1.0, q2=np.radians(0), q3=np.radians(90))
    # plot_robot(q1=1.0, q2=np.radians(20), q3=np.radians(90))
    # plot_robot(q1=1.0, q2=np.radians(40), q3=np.radians(90))
    
    # # # C2
    # data = workspace_demo()
    # points = data[:,:3]
    # joint_values = data[:, 3:]
    
    # # # C3
    # # p_goal = np.array([1.2, -1.5, 2.0])
    # # q1, q2, q3, d = ikine(p_goal,points, joint_values)

    # # print(f"q1={q1:.2f}, q2={np.degrees(q2):.1f}°, q3={np.degrees(q3):.1f}°, σφάλμα={d:.4f} m")
    # # plot_robot(q1, q2, q3)
    
    
    # # C4
    # helper_plot([0, -1, -1], [2, -1, 1.0], 5.0, points, joint_values, order=5)
    # helper_plot([1, 0, 0], [1.0, np.radians(0), np.radians(90)], 5.0, points, joint_values, order=5,space="work")
    pass