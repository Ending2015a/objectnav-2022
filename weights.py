"""
  0. void: 0.058571
  1. wall: 0.417352
  2. floor: 0.089779
  3. chair: 0.017919
  4. door: 0.104468
  5. table: 0.010811
  6. picture: 0.010412
  7. cabinet: 0.036080
  8. cushion: 0.005023
  9. window: 0.034773
  10. sofa: 0.014949
  11. bed: 0.013166
  12. curtain: 0.015915
  13. chest_of_drawers: 0.005356
  14. plant: 0.002068
  15. sink: 0.000687
  16. stairs: 0.013527
  17. ceiling: 0.061818
  18. toilet: 0.001625
  19. stool: 0.000682
  20. towel: 0.001256
  21. mirror: 0.006101
  22. tv_monitor: 0.006078
  23. shower: 0.001915
  24. column: 0.001536
  25. bathtub: 0.001014
  26. counter: 0.003201
  27. fireplace: 0.002126
  28. lighting: 0.002945
  29. beam: 0.001688
  30. railing: 0.007162
  31. shelving: 0.004767
  32. blinds: 0.002011
  33. gym_equipment: 0.000278
  34. seating: 0.001264
  35. board_panel: 0.000469
  36. furniture: 0.011130
  37. appliances: 0.004479
  38. clothes: 0.001305
  39. objects: 0.024291
"""



l = [0.8868035078048706, 0.8125827312469482, 0.8695287108421326, 0.9385072588920593, 0.8635608553886414, 0.9624459743499756, 0.9642767906188965, 0.9072492718696594, 1.0011916160583496, 0.9088456630706787, 0.9469540119171143, 0.9529639482498169, 0.9440186619758606, 0.9978256821632385, 1.0501205921173096, 1.1180226802825928, 0.9516791105270386, 0.8845832943916321, 1.064274549484253, 1.118451476097107, 1.0797929763793945, 0.9910647869110107, 0.9912649989128113, 1.054595947265625, 1.0676296949386597, 1.0930585861206055, 1.0254521369934082, 1.0485239028930664, 1.0300689935684204, 1.0619986057281494, 0.9828771352767944, 1.0039446353912354, 1.0517538785934448, 1.1807304620742798, 1.079397201538086, 1.1435633897781372, 0.9610335826873779, 1.0072451829910278, 1.0774619579315186, 0.9246569275856018]


print('[' + ', '.join(['{:.6f}'.format(f) for f in l]) + ']')