import chess
import numpy as np


class FogOfWarChessBoard(chess.Board):
    def __init__(self, fen=None):
        if fen:
            super().__init__(fen=fen)
        else:
            super().__init__()

    # def is_variant_end(self) -> bool:
    #     return self.is_variant_win() or self.is_variant_loss()

    def is_variant_draw(self) -> bool:
        return False

    def is_variant_loss(self) -> bool:
        return not self.king(self.turn)

    def is_variant_win(self) -> bool:
        return not self.king(not self.turn)

    def is_insufficient_material(self) -> bool:
        return False

    def get_canonical_board(self, p_color):
        if not p_color:
            return self.copy().mirror()
        return self

    def is_legal(self, move: chess.Move) -> bool:
        return not self.is_variant_end() and self.is_pseudo_legal(move)

    # def _castling_uncovers_rank_attack(self, rook_bb, king_to):
    #     return False

    def was_into_check(self):
        return False

    def is_check(self):
        return False

    def is_checkmate(self):
        return False

    def is_stalemate(self) -> bool:
        return False

    def is_attacked_by(self, color, square):
        if color == self.turn:
            return super().is_attacked_by(color, square)
        else:
            return False

    def generate_castling_moves(self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL, *, connected_kings=False, only_true_castling=False):
        return super().generate_castling_moves(from_mask, to_mask)

    def push_action(self, action):
        from utils import action_to_move
        move = action_to_move(action)
        if move in self.pseudo_legal_moves:
            self.push(move)
        else:
            raise ValueError(f"Invalid action: {action}")

    # OMITTED BECAUSE IT GIVES UNFAIR INFORMATION
    # def get_repetition_planes(self):
    #
    #     # Create the repetition planes
    #     first_repetition_plane = np.full((8, 8), self.is_repetition(1))
    #     second_repetition_plane = np.full((8, 8), self.is_repetition(2))
    #
    #     return np.stack([first_repetition_plane, second_repetition_plane])

    # OMITTED FOR NOW DUE TO UNCERTAINTY
    # def get_known_castling_rights(self, player_color):
    #     king_square = chess.E1 if player_color == chess.WHITE else chess.E8
    #     king_start = self.king(player_color ^ 1)
    #     if king_start is not None and king_start != king_square:
    #         return {"kside": False, "qside": False}
    #
    #     known_rights = {"kside": False, "qside": False}
    #     for side in [chess.A1, chess.H1]:
    #         opposite_side = side ^ 0x38
    #         rook_square = side if player_color == chess.WHITE else opposite_side
    #         rook_start = self.piece_at(rook_square)
    #
    #         if rook_start is not None and rook_start.piece_type == chess.ROOK and rook_start.color == (
    #                 player_color ^ 1):
    #             if side == chess.A1:
    #                 known_rights["qside"] = True
    #             else:
    #                 known_rights["kside"] = True
    #
    #     return known_rights
