基于用户行为数据，训练一个推荐模型，用于推荐二手设备。


编写python脚本，帮我在mysql中创建一张表，并生成50000条不连续的userid和itemid以及rating。但用户分布整体满足generate_user_behavior_data函数的用户分布，方便我后续验证。



### MySQL表结构
```sql
CREATE TABLE user_behavior (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(20) NOT NULL,
    item_id VARCHAR(20) NOT NULL,
    rating DECIMAL(3,2) NOT NULL,
    user_type VARCHAR(20) NOT NULL,
    item_category VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_item_id (item_id),
    INDEX idx_user_type (user_type),
    INDEX idx_item_category (item_category)
);


mysql配置信息如下：
DB_HOST=localhost
DB_PORT=3306
DB_USERNAME=test
DB_PASSWORD=test
DB_DATABASE=testdb
