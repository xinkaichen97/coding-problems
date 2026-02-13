"""
Problems for Heaps / Priority Queues
"""
import heapq


class KthLargest:
    """
    https://neetcode.io/problems/kth-largest-integer-in-a-stream
    Time: O(logk) for each add() call, Space: O(k)
    """
    
    def __init__(self, k: int, nums: List[int]):
        
        self.nums = nums
        self.k = k
        heapq.heapify(self.nums)
        # remove the smallest (n - k) items
        while len(self.nums) > k:
            heapq.heappop(self.nums)

    def add(self, val: int) -> int:
        heapq.heappush(self.nums, val)
        # maintain size of k
        if len(self.nums) > self.k:
            heapq.heappop(self.nums)
        # the smallest in the heap is the k-th largest
        return self.nums[0]


class Solution:

    def lastStoneWeight(self, stones: List[int]) -> int:
        """
        https://neetcode.io/problems/last-stone-weight
        Time: O(nlogn), Space: O(n)
        """
        # reverse stones to create a MaxHeap
        stones = [-s for s in stones]
        heapq.heapify(stones)

        # keep popping and pushing until the heap has more than one item
        while len(stones) > 1:
            # push the biggest stones (now negative)
            first = heapq.heappop(stones)
            second = heapq.heappop(stones)
            # second is either equal to or bigger than first
            # if not equal, push the difference (also negative)
            if second > first:
                heapq.heappush(stones, first - second)

        # if not empty, return the last one in the heap
        return abs(stones[0]) if stones else 0
        
    
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        https://neetcode.io/problems/k-closest-points-to-origin
        Time: O(n * logk), Space: O(k)
        """
        heap = []
        for x, y in points:
            # no need to do math.sqrt()
            dist = x ** 2 + y ** 2
            # negate distance to create a MaxHeap - better time/space complexity than a MinHeap
            # also add the coordinates
            heapq.heappush(heap, [-dist, x, y])
            # keep only k items
            if len(heap) > k:
                heapq.heappop(heap)

        res = []
        # pop every item from the MaxHeap
        while heap:
            dist, x, y = heapq.heappop(heap)
            res.append([x, y])
        return res


    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        https://neetcode.io/problems/kth-largest-element-in-an-array
        Time: O(nlogk), Space: O(k)
        """
        heap = []
        for num in nums:
            # keep a MinHeap of size k
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        # return heapq.nlargest(k, nums)[-1]
        return heap[0]


    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        https://neetcode.io/problems/task-scheduling
        Time: O(m), Space: O(1), m - len(tasks)
        """
        # create a MaxHeap for frequencies
        counts = Counter(tasks)
        heap = [-count for count in counts.values()]
        heapq.heapify(heap)

        # create a queue for cooldown (-count, freeTime)
        q = deque()
        time = 0
        # iterate until both heap and queue are empty
        while heap or q:
            time += 1
            # decrease count and add to queue if not zero
            if heap:
                count = 1 + heapq.heappop(heap)
                if count != 0:
                    q.append((count, time + n))
            else:
                # optional - fast forward to the next freeTime
                time = q[0][1]
            # if one is free, pop from the queue and add to the heap
            if q and time == q[0][1]:
                heapq.heappush(heap, q.popleft()[0])
            
        return time


class Twitter:
    """
    https://neetcode.io/problems/design-twitter-feed
    Time: O(nlogn) for each getNewsFeed(), Space: O(N * m + N * M + n)
    n - # of followeeId for current userId, m - max # of tweets by a user
    N - # of userIds, M - max # of followees
    """
    def __init__(self):
        self.count = 0
        self.tweetMap = defaultdict(list)
        self.followMap = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        # add count, tweetId to tweetMap[userId]
        # use negative count for keeping the latest one (higher absolute value) on top of the heap
        self.tweetMap[userId].append([self.count, tweetId])
        # keep only 10 tweets for each userId
        if len(self.tweetMap[userId]) > 10:
            self.tweetMap[userId].pop(0)
        self.count -= 1

    def getNewsFeed(self, userId: int) -> List[int]:
        res = []
        heap = []
        # add user to their own followMap to display their own tweets
        self.followMap[userId].add(userId)

        # go through each followee
        for followeeId in self.followMap[userId]:
            # push to heap
            if followeeId in self.tweetMap:
                index = len(self.tweetMap[followeeId]) - 1
                count, tweetId = self.tweetMap[followeeId][index]
                heapq.heappush(heap, [count, tweetId, followeeId, index - 1])

        # add the 10 most recent tweets to the results
        while heap and len(res) < 10:
            count, tweetId, followeeId, index = heapq.heappop(heap)
            res.append(tweetId)
            if index >= 0:
                count, tweetId = self.tweetMap[followeeId][index]
                heapq.heappush(heap, [count, tweetId, followeeId, index - 1])
                
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        # add to followMap[followerId]
        if followerId != followeeId:
            self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # remove from followMap[followerId]
        if followeeId in self.followMap[followerId]:
            self.followMap[followerId].remove(followeeId)


class MedianFinder:
    """
    https://neetcode.io/problems/find-median-in-a-data-stream
    Time: addNum - O(m * logn), findMedian - O(m), Space: O(n)
    m - # of function calls, n - len(array)
    """
    
    def __init__(self):
        # maintain two lists to be converted into heaps,
        # small - Max Heap (negate each item), large - Min Heap
        # push and pop - both O(logn)
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        # if num is greater than the min of large, push to large
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        # otherwise push to small (times -1 to maintain a max heap)
        else:
            heapq.heappush(self.small, -1 * num)

        # rebalance heaps
        # if large has too many elements, pop the smallest and push to small
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)
        # if small has too many elements, pop the largest and push to large
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)

    def findMedian(self) -> float:
        # if odd length and large has more, the median is the smallest in large
        if len(self.large) > len(self.small):
            return self.large[0]
        # if small has more, the median is the largest in small
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        # even length, calculate the mean
        return (-1 * self.small[0] + self.large[0]) / 2.0
        
