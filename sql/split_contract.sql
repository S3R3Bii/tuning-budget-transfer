SELECT group_id, COUNT(DISTINCT split) AS partitions
FROM assignments
GROUP BY group_id
HAVING COUNT(DISTINCT split) <> 1;
