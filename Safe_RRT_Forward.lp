\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 Control_Angular_Velocity ^2 ] / 2 
Subject To
 CBF_constraint: - 0.9365678911411374 Control_Angular_Velocity
   >= 1.783870924627305
 CBF_constraint: 0.4765348917448175 Control_Angular_Velocity
   >= 1.982957081925183
 CBF_constraint: - 3.467883696149977 Control_Angular_Velocity
   >= -0.9665651076653719
 CBF_constraint: - 1.803538815178632 Control_Angular_Velocity
   >= 1.014729647710962
Bounds
 -4.25 <= Control_Angular_Velocity <= 4.25
End
