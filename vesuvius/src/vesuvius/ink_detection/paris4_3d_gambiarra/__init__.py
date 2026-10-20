"""Training and dense inference for the PHercParis4 3D ink model ink_paris4_3d_4.8um_gambiarra.

Teacher: a DINOv2-style 3D ViT (dinovol_2) fine-tuned on labeled points in four stages.
Student: a vesuvius NetworkFromConfig U-Net on 4.8 um input, distilled from the teacher and the points in two stages.
See README.md for the recipe and commands.
"""
