\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 Control_Translation_Velocity ^2 + 2 Control_Angular_Velocity ^2 ] / 2 
Subject To
 CLF_constraint: - 4.341913490069015 Control_Translation_Velocity
   + 1.417361478688486 Control_Angular_Velocity <= -0.8573492192102692
Bounds
 -0.45 <= Control_Translation_Velocity <= 0.45
 -4.25 <= Control_Angular_Velocity <= 4.25
End
