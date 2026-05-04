此demo文件夹基于原版demo进行重构，旨在解耦出共享的部分。
重构后的结构如下：

- vlm
    - vlm client
    - vlm 功能函数
    - vlm test # 用于测试 vlm 功能函数

- skills
    - pnp_skill
        - pick_and_place.py
        - config.yaml

    - air_fryer_skill
        - air_fryer.py
    
    - tools
        - atomic_actions.py
        - config_utils.py
        - utils.py

- task
    - pnp
        - run.py
        - config.yaml
        
    - roast_sweet_potatoes
        - run.py
        - config.yaml
        

- gradio    # 运行演示界面脚本
    - run   # 需要能够方便的增删task的结构

- logs