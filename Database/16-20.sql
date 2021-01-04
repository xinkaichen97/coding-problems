-- Problem 16 (No. 601)
-- https://leetcode.com/problems/human-traffic-of-stadium/

SELECT id AS "id", TO_CHAR(visit_date, 'YYYY-MM-DD') AS "visit_date", people AS "people"
FROM stadium
WHERE people >= 100
AND 
((id + 1 IN (SELECT id FROM stadium WHERE people >= 100) AND id + 2 IN (SELECT id FROM stadium WHERE people >= 100))
OR (id + 1 IN (SELECT id FROM stadium WHERE people >= 100) AND id - 1 IN (SELECT id FROM stadium WHERE people >= 100))
OR (id - 1 IN (SELECT id FROM stadium WHERE people >= 100) AND id - 2 IN (SELECT id FROM stadium WHERE people >= 100)));
                                                                          
-- Alternative
SELECT DISTINCT t1.id AS "id", TO_CHAR(t1.visit_date, 'YYYY-MM-DD') AS "visit_date", t1.people AS "people"
FROM stadium t1, stadium t2, stadium t3
WHERE t1.people >= 100 AND t2.people >= 100 AND t3.people >= 100
AND
((t1.id - t2.id = 1 AND t1.id - t3.id = 2 AND t2.id - t3.id = 1)  -- t1, t2, t3
OR (t2.id - t1.id = 1 AND t2.id - t3.id = 2 AND t1.id - t3.id = 1) -- t2, t1, t3
OR (t3.id - t2.id = 1 AND t2.id - t1.id = 1 AND t3.id - t1.id = 2)) -- t3, t2, t1
ORDER BY t1.id;
    

-- Problem 17 (No. 620)
-- https://leetcode.com/problems/not-boring-movies/

SELECT * 
FROM cinema
WHERE description <> 'boring'
AND MOD(id, 2) = 1
ORDER BY rating DESC


-- Problem 18 (No. 626)
-- https://leetcode.com/problems/exchange-seats/
SELECT (CASE WHEN MOD(id, 2) != 0 AND seat_count != id THEN id + 1
        WHEN MOD(id, 2) != 0 AND seat_count = id THEN id 
        ELSE id - 1 END) AS id, student
FROM seat, (SELECT COUNT(*) AS seat_count FROM seat)
ORDER BY id;


-- Problem 19 (No. 627)
-- https://leetcode.com/problems/swap-salary/
UPDATE salary
SET sex = CASE WHEN sex = 'f' THEN 'm' ELSE 'f' END;
