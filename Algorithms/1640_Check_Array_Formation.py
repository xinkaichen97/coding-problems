class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        # initialize piece positions using first elements in pieces (because the order cannot be changed within a piece)
        piece_dict = {piece[0]: i for i, piece in enumerate(pieces)}
        i = 0
        while i < len(arr):
            if arr[i] not in piece_dict:
                return False
            j = 0
            # find the position in pieces
            pos = piece_dict[arr[i]]
            # loop through every number in this piece to check if there's match
            while j < len(pieces[pos]):
                if arr[i] != pieces[pos][j]:
                    return False
                i += 1 
                j += 1
        return True
