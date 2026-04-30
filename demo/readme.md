demo文件夹下：
1. pick_and_place_old：年前的pnp老脚本。需要和 moveit control端一起完成任务。
2. pnp_final：实现了多线程连续pnp，使用重新封好的realman env。使用/pnp_final/pick_and_place.py脚本可以通过修改指令来完成pnp任务。使用pnp_final/pnp_gradio_demo.py脚本可以启动人机交互界面来控制机械臂完成任务。
3. pnp_final_test：在pnp_final的基础上改进，使用带方位描述的目标进行打点，脚本已经基本改好，prompt可能还需要调。
4. use_air_fryer：空气炸锅demo
5. 任务演示界面