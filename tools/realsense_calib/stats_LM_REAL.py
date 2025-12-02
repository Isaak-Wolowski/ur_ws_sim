import numpy as np
import helper_functions as util
import os 
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import check_grad
from functools import partial
import time
import winsound

dir_path = os.path.realpath(os.path.dirname(__file__))




#np.random.seed(45)


# Pose Distribution
start_index = 0
end_index = 220

no_of_runs = 100



rot_errorsX_runs = []
rot_errorsY_runs = []
rot_errorsZ_runs = []
rot_errors_total_runs = []

trans_errorsX_runs = []
trans_errorsY_runs = []
trans_errorsZ_runs = []
trans_errors_total_runs = []

loop_closure_errors_R = []
loop_closure_errors_t = []



# Execution time
start = time.time()  # tic

for run in range(no_of_runs):
    ###############################################################################
    ###############################################################################
    ####                     1. INITIALIZATION/ GT CALCULATION                  ###
    ###############################################################################
    ###############################################################################

    # 1.1 Retrieve pose data already saved in a directory  
    #------------------------------------------------------------------------------    

    file_path_A = f"{dir_path}/data/A_poses.npy"        # S1 wrt base
    file_path_B = f"{dir_path}/data/B_poses.npy"        # target wrt S2 
    file_path_C = f"{dir_path}/data/C_poses.npy"        # target wrt S3




    # 1.2 Load absolute real poses from files and decompose them into rmats and tvecs
    #------------------------------------------------------------------------------
    # tcp poses
    A_poses = util.load_and_slice_poses(file_path_A, start_index, end_index)
    A_rmats, A_tvecs = util.pose_decomposition_list(A_poses)

    # I. Retrieve data of camera poses (visaul target with respect to camera)
    B_poses = util.load_and_slice_poses(file_path_B, start_index, end_index)
    B_rmats, B_tvecs = util.pose_decomposition_list(B_poses)

    # II. Retrieve data of camera poses (visaul target with respect to camera)
    C_poses = util.load_and_slice_poses(file_path_C, start_index, end_index)
    C_rmats, C_tvecs = util.pose_decomposition_list(C_poses)


        


    ###############################################################################
    ###############################################################################
    ####              2. PREPARE MEASUREMENT DATA:  Ai, Bi and Ci               ###
    ###############################################################################
    ###############################################################################


    # 2.3 Calculate noisy poses change and decompose into R and t  
    # opencv: A = inv(A1)@A2, B = B1@inv(B2) C = C1@inv(C2)
    #------------------------------------------------------------------------------      
    Ai = []
    for i in range(len(A_poses)-1):
        Ai.append((util.pose_inverse(A_poses[i]))@A_poses[i+1])

    R_Ai, t_Ai = util.pose_decomposition_list(Ai)


    Bi = []                     
    for i in range(len(B_poses)-1):
        Bi.append((B_poses[i]@util.pose_inverse(B_poses[i+1])))
        
    R_Bi, t_Bi = util.pose_decomposition_list(Bi)

        
    Ci = []                     
    for i in range(len(C_poses)-1):
        Ci.append(C_poses[i]@(util.pose_inverse(C_poses[i+1])))

    R_Ci, t_Ci = util.pose_decomposition_list(Ci)


    # Initialize the optimization with tsai and lenz method
    RX, tX = cv.calibrateHandEye(A_rmats, A_tvecs, B_rmats, B_tvecs, method=cv.CALIB_HAND_EYE_TSAI)
    RY, tY = cv.calibrateHandEye(A_rmats, A_tvecs, C_rmats, C_tvecs, method=cv.CALIB_HAND_EYE_TSAI)
    RZ, tZ = cv.calibrateHandEye(B_rmats, B_tvecs, C_rmats, C_tvecs, method=cv.CALIB_HAND_EYE_TSAI)

    sX = util.rmat_to_rod(RX) 
    sY = util.rmat_to_rod(RY)
    sZ = util.rmat_to_rod(RZ)


    # # Plot absolute poses
    # util.plot_one_set_of_poses(A_poses)
    # util.plot_one_set_of_poses(B_poses)
    # util.plot_one_set_of_poses(C_poses)

    # # Plot relative poses
    # util.plot_one_set_of_poses(Ai)
    # util.plot_one_set_of_poses(Bi)
    # util.plot_one_set_of_poses(Ci






    ###############################################################################
    #######                SLSQP  Using analytical dirivatives              #######
    ###############################################################################

    #------------------------------------------------------------------------------
    # HARD CONSTRAINT: LAGRANGIAN MULTIPLIER
    def constraint_R(s):
        sX, sY, sZ = s[:3], s[3:6], s[6:9]
        RX = util.rod_to_rmat(sX)
        RY = util.rod_to_rmat(sY)
        RZ = util.rod_to_rmat(sZ)
        sC = util.rmat_to_rod(RX @ RY) - util.rmat_to_rod(RZ)
        return sC.flatten()





    # def gradient_fxn_constraint_R(s, lambdah):
    #     sX, sY, sZ = s[:3], s[3:6], s[6:9]
        
    #     RX = util.rod_to_rmat(sX)
    #     RY = util.rod_to_rmat(sY)
    #     RZ = util.rod_to_rmat(sZ)
        
    #     conX = util.constraint_gradX(RX, RY, RZ, lambdah)
    #     conY = util.constraint_gradY(RX, RY, RZ, lambdah)
    #     conZ = util.constraint_gradZ(RX, RY, RZ, lambdah)
        
    #     # Combine gradients into a single array
    #     con = np.hstack([conX, conY, conZ])
    #     return con
    #------------------------------------------------------------------------------

    def obj_func_R(s, R_A, R_B):
        R_X = util.rod_to_rmat(s)
        loss = sum([np.linalg.norm(RA @ R_X - R_X @ RB)**2 for RA, RB in zip(R_A, R_B)])
        return (1 / (2 * len(R_A))) * loss

    def obj_func_R_overall(s, R_Ai, R_Bi, R_Ci):

        sX, sY, sZ = s[:3], s[3:6], s[6:9]
        
        # Compute losses for RX, RY, RZ
        lossRX = obj_func_R(s=sX, R_A=R_Ai , R_B=R_Bi)
        lossRY = obj_func_R(s=sY, R_A=R_Bi , R_B=R_Ci)
        lossRZ = obj_func_R(s=sZ, R_A=R_Ai , R_B=R_Ci)
        
        # Total loss
        loss = lossRX + lossRY + lossRZ
        return loss 


    # def gradient_R(R_X, R_A, R_B):
        
    #     # Compute partial derivatives of RX with respect to s (dR_ds)
    #     dR_ds0, dR_ds1, dR_ds2, _, _, _ = util.dR_ds(R_X)
            
    #     # Compute gradients for each parameter using dLR_ds
    #     grad0 = util.dLR_ds(R_A, R_B, R_X, dR_ds0)
    #     grad1 = util.dLR_ds(R_A, R_B, R_X, dR_ds1)
    #     grad2 = util.dLR_ds(R_A, R_B, R_X, dR_ds2)
        
    #     # Combine gradients into a single array
    #     grad = np.array([grad0, grad1, grad2]) 
    #     #print('Analytical gradients used')
    #     return grad

    # def gradient_R_overall(s, R_Ai, R_Bi, R_Ci, lambdah):
    #     sX, sY, sZ = s[:3], s[3:6], s[6:9]
        
    #     RX = util.rod_to_rmat(sX)
    #     RY = util.rod_to_rmat(sY)
    #     RZ = util.rod_to_rmat(sZ)
        
    #     gradX = gradient_R(R_X=RX, R_A=R_Ai, R_B=R_Bi)
    #     gradY = gradient_R(R_X=RY, R_A=R_Bi, R_B=R_Ci)
    #     gradZ = gradient_R(R_X=RZ, R_A=R_Ai, R_B=R_Ci)

        
    #     # Combine gradients into a single array
    #     grad_overall = np.hstack([gradX, gradY, gradZ])
    #     #print(grad_overall)
    #     return grad_overall


    def callback(s):
        global R_Ai, R_Bi, R_Ci
        #print(f"Callback received s: shape={s.shape}, s={s}")
        sX, sY, sZ = s[:3], s[3:6], s[6:9]
        #print(sX)
        RX = util.rod_to_rmat(sX)
        RY = util.rod_to_rmat(sY) 
        RZ = util.rod_to_rmat(sZ) 
        
        loss = obj_func_R_overall(s, R_Ai, R_Bi, R_Ci)
        losses_total.append(loss)
        
        rot_errorsX.append(util.compute_eR1(RX, R_Ai, R_Bi))
        rot_errorsY.append(util.compute_eR1(RY, R_Bi, R_Ci))
        rot_errorsZ.append(util.compute_eR1(RZ, R_Ai, R_Ci))
        rot_errors_total.append(util.compute_eR1(RX, R_Ai, R_Bi) + util.compute_eR1(RY, R_Bi, R_Ci) + util.compute_eR1(RZ, R_Ai, R_Ci))
        
        RXs.append(RX)
        RYs.append(RY)
        RZs.append(RZ)
        
           
        
    
        
    ###############################################################################
    ##  -------------------------- Rotations optimization --------------------------

    losses_total = []
    rot_errorsX = []
    rot_errorsY = []
    rot_errorsZ = []
    rot_errors_total = []
    RXs = []
    RYs = []
    RZs = []


    # 1. Optimize rotations
    s = np.random.randn(9)

    # Define constraint dictionary   
    constraint_dict = {
        'type': 'eq',  # Equality constraint
        'fun': constraint_R,  # Constraint function
        #'args': (lambdah,),  
        #'jac': gradient_fxn_constraint_R
        }

    # Optimization bounds (optional)
    bounds = [(-np.pi, np.pi)] * 9  # Bounds for Rodrigues vectors

    # 1. Optimize rotations using SLSQP
    result = minimize(
        obj_func_R_overall,
        x0 = s,
        args=(R_Ai, R_Bi, R_Ci),
        #jac = gradient_R_overall,
        method = 'SLSQP',
        bounds = bounds,
        callback = callback,
        constraints=[constraint_dict],  # List of constraints
        options={
            'disp': False,
            'ftol': 1e-18,   # Function tolerance (small value means higher precision)
            'gtol': 1e-6,    # Gradient tolerance (small value means higher precision)
            'eps': 1e-6,     # Step size for numerical gradient approximation (if needed)
            'maxiter': 100, # Maximum number of iterations
        }
    )
    
    # Extract optimized Rodrigues vectors
    sX_opt, sY_opt, sZ_opt = result.x[:3], result.x[3:6], result.x[6:9]
    
    rot_errorsX_runs.append(rot_errorsX[-1])
    rot_errorsY_runs.append(rot_errorsY[-1])
    rot_errorsZ_runs.append(rot_errorsZ[-1])
    rot_errors_total_runs.append(rot_errors_total[-1])
    
    
    RX_opt = util.rod_to_rmat(sX_opt)
    RY_opt = util.rod_to_rmat(sY_opt)
    RZ_opt = util.rod_to_rmat(sZ_opt)

    constraint_R_error = np.linalg.norm(RX_opt @ RY_opt - RZ_opt)
    loop_closure_errors_R.append(constraint_R_error)
    
        

        



    ###############################################################################
    ##  -------------------------- Translations optimization ----------------------
    
    ### Lagrangian Multiplier (LM) Constrained Solution
    tX_opt, tY_opt, tZ_opt, lambda_vec = util.solve_constrained_lsq_LM(R_Ai, 
                                                              R_Bi, 
                                                              RX_opt, 
                                                              RY_opt, 
                                                              RZ_opt, 
                                                              t_Ai, 
                                                              t_Bi, 
                                                              t_Ci, 
                                                              reg=1e-12)
    

    #tX_opt, tY_opt, tZ_opt = (t[:3]).reshape(3,1), (t[3:6]).reshape(3,1), (t[6:9]).reshape(3,1)

    
    constraint_t_error = np.linalg.norm(RX_opt @ tY_opt + tX_opt - tZ_opt)
    loop_closure_errors_t.append(constraint_t_error)
    
    # print(f"Loop closure Rot. error :{constraint_R_error}")
    #print(f"Loop closure Trans. error :{constraint_t_error}")

    
    trans_errorsX_runs.append(util.compute_et1(R_Ai, t_Ai, t_Bi, tX_opt, RX_opt))
    trans_errorsY_runs.append(util.compute_et1(R_Bi, t_Bi, t_Ci, tY_opt, RY_opt))
    trans_errorsZ_runs.append(util.compute_et1(R_Ai, t_Ai, t_Ci, tZ_opt, RZ_opt))
    trans_errors_total_runs.append(util.compute_et1(R_Ai, t_Ai, t_Bi, tX_opt, RX_opt) + util.compute_et1(R_Bi, t_Bi, t_Ci, tY_opt, RY_opt) + util.compute_et1(R_Ai, t_Ai, t_Ci, tZ_opt, RZ_opt))
    
    
    
    print(f"Completed rot and trans run :{run+1}")
        
    #--------------------------------------------------------------------------





   
# Save rotation errors in csv   
util.outlier_removal_ModifiedZ_and_stats(rot_errorsX_runs, csv_filename='LM_rotX_statistics.csv', threshold=3.5)
util.outlier_removal_ModifiedZ_and_stats(rot_errorsY_runs, csv_filename='LM_rotY_statistics.csv', threshold=3.5)
util.outlier_removal_ModifiedZ_and_stats(rot_errorsZ_runs, csv_filename='LM_rotZ_statistics.csv', threshold=3.5)
filtered_data_rotTot, mean_final_rotTot, median_final_rotTot, std_dev_final_rotTot = util.outlier_removal_ModifiedZ_and_stats(rot_errors_total_runs, csv_filename='LM_rotTot_statistics.csv', threshold=3.5)

# Save translation errors in csv(closed-form solution)   
util.outlier_removal_ModifiedZ_and_stats(trans_errorsX_runs, csv_filename='LM_transX_statistics.csv', threshold=3.5)
util.outlier_removal_ModifiedZ_and_stats(trans_errorsY_runs, csv_filename='LM_transY_statistics.csv', threshold=3.5)
util.outlier_removal_ModifiedZ_and_stats(trans_errorsZ_runs, csv_filename='LM_transZ_statistics.csv', threshold=3.5)
filtered_data_transTot, mean_final_transTot, median_final_transTot, std_dev_final_transTot = util.outlier_removal_ModifiedZ_and_stats(trans_errors_total_runs, csv_filename='LM_transTot_statistics.csv', threshold=3.5)


filtered_data_constraintError_R, _, _, _ = util.outlier_removal_ModifiedZ_and_stats(loop_closure_errors_R, csv_filename='LM_constraintError_R_statistics.csv', threshold=3.5)
filtered_data_constraintError_t, _, _, _ = util.outlier_removal_ModifiedZ_and_stats(loop_closure_errors_t, csv_filename='LM_constraintError_t_statistics.csv', threshold=3.5)

util.save_list_to_csv("filtered_data_rotTot.csv", filtered_data_rotTot, column_name=end_index)
util.save_list_to_csv("filtered_data_transTot.csv", filtered_data_transTot, column_name=end_index)
util.save_list_to_csv("filtered_data_constraintError_R.csv", filtered_data_constraintError_R, column_name=end_index)
util.save_list_to_csv("filtered_data_constraintError_t.csv", filtered_data_constraintError_t, column_name=end_index)

end = time.time()    # toc
print(f"Elapsed time: {(end - start)/60:.6f} minutes")
winsound.Beep(1000, 5000)
