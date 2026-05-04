此demo文件夹基于原版demo进行重构，旨在解耦出共享的部分。
重构后的结构如下：
- vlm
    - vlm client
    - vlm 功能函数
- skill
    - pnp # 将单次 pnp 封成原子动作
        - pnp 功能函数
        - pnp 核心代码
    - 
- task
    - pnp
        - config
        - run
    - air_fryer
        - config
        - run
- gradio    # 运行演示界面脚本
    - run   # 需要能够方便的增删task的结构
- logs