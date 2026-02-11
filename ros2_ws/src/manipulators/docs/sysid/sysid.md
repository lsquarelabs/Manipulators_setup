System Identification Options                                                                                                                                                               
                                                                                                                                                                                              
  Given your goal of accurate gravity compensation and dynamic motions, here are the viable approaches:                                                                                       
                                                                                                                                                                                              
  Option 1: Regressor-Based Inverse Dynamics (Recommended)                                                                                                                                    
                                                                                                                                                                                              
  The standard and most powerful method. Uses the linear relationship:                                                                                                                        
                                                                                                                                                                                              
  τ = Y(q, q̇, q̈) · π                                                                                                                                                                          
                                                                                                                                                                                              
  where Y is the regressor matrix and π is the vector of dynamic parameters (mass, CoM, inertia, friction per joint).                                                                         
                                                                                                                                                                                              
  - Pinocchio has computeJointTorqueRegressor() which gives you Y directly                                                                                                                    
  - Solve via least-squares (OLS, WLS, or regularized)                                                                                                                                        
  - Identifies all identifiable inertial + friction parameters                                                                                                                                
  - ~10 inertial params/link × 7 links + friction params, reduced to identifiable base parameters                                                                                             
  - Requires exciting trajectories (Fourier-parameterized sinusoids)                                                                                                                          
  - Best for: gravity comp, feedforward torque, full model accuracy                                                                                                                           
                                                                                                                                                                                              
  Option 2: Gravity-Only Identification                                                                                                                                                       
                                                                                                                                                                                              
  A simplified subset of Option 1 that only identifies parameters affecting gravity torques (mass × CoM products per link).                                                                   
                                                                                                                                                                                              
  - Record static/quasi-static poses across the workspace                                                                                                                                     
  - Only need q and τ (no velocity/acceleration needed)                                                                                                                                       
  - Much simpler data collection (hold poses, record)                                                                                                                                         
  - ~3-4 identifiable params per link                                                                                                                                                         
  - Best for: gravity compensation only, quick to set up                                                                                                                                      
                                                                                                                                                                                              
  Option 3: Energy-Based / Power-Based Method                                                                                                                                                 
                                                                                                                                                                                              
  Uses the power equation P = τᵀq̇ integrated over time instead of instantaneous dynamics.                                                                                                     
                                                                                                                                                                                              
  - Avoids computing q̈ (acceleration), which is noisy                                                                                                                                         
  - More robust to measurement noise                                                                                                                                                          
  - Same exciting trajectories as Option 1                                                                                                                                                    
  - Best for: when acceleration estimation is unreliable                                                                                                                                      
                                                                                                                                                                                              
  Option 4: Data-Driven (Neural Network)                                                                                                                                                      
                                                                                                                                                                                              
  Train a network to predict τ from (q, q̇, q̈) or use a physics-informed architecture.                                                                                                         
                                                                                                                                                                                              
  - Can capture unmodeled nonlinearities (cable routing, joint flexibility)                                                                                                                   
  - Less interpretable, needs more data                                                                                                                                                       
  - Harder to validate physically                                                                                                                                                             
  - Best for: capturing residual dynamics after a model-based approach    