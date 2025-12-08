"""
Makerspace RAG - User Model
Admin user authentication
"""

from flask_login import UserMixin


class AdminUser(UserMixin):
    """Simple admin user for authentication."""

    def __init__(self, user_id):
        self.id = user_id
