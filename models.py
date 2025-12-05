"""
Makerspace RAG - Database Models
Simple SQLite database for component tracking
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean

db = SQLAlchemy()


class Component(db.Model):
    """A component in the makerspace inventory"""
    __tablename__ = 'components'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hylleplass: Mapped[str] = mapped_column(String(50), nullable=False)  # Always uppercase
    forbruksvare: Mapped[bool] = mapped_column(Boolean, default=False)
    restock: Mapped[bool] = mapped_column(Boolean, default=False)
    antall: Mapped[int] = mapped_column(Integer, default=0)
    
    def __repr__(self):
        return f'<Component {self.name} @ {self.hylleplass}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'hylleplass': self.hylleplass,
            'forbruksvare': self.forbruksvare,
            'restock': self.restock,
            'antall': self.antall
        }


# =============================================================================
# HELPER FUNCTIONS FOR AI QUERIES
# =============================================================================

def search_components_db(query: str, limit: int = 20):
    """Search components by name or hylleplass"""
    query_upper = query.upper()
    query_lower = query.lower()
    
    results = Component.query.filter(
        db.or_(
            Component.name.ilike(f'%{query_lower}%'),
            Component.hylleplass.ilike(f'%{query_upper}%')
        )
    ).limit(limit).all()
    
    return results


def get_components_at_location(hylleplass: str):
    """Get all components at a specific hylleplass"""
    return Component.query.filter(
        Component.hylleplass == hylleplass.upper()
    ).all()


def get_restock_needed():
    """Get all components that need restocking"""
    return Component.query.filter(Component.restock == True).all()


def get_forbruksvarer():
    """Get all consumables"""
    return Component.query.filter(Component.forbruksvare == True).all()


def add_component(name: str, hylleplass: str, forbruksvare: bool = False, 
                  restock: bool = False, antall: int = 0):
    """Add a new component - hylleplass auto-uppercased"""
    component = Component(
        name=name,
        hylleplass=hylleplass.upper(),  # ENFORCE CAPS
        forbruksvare=forbruksvare,
        restock=restock,
        antall=antall
    )
    db.session.add(component)
    db.session.commit()
    return component


def update_component(component_id: int, **kwargs):
    """Update a component - hylleplass auto-uppercased if provided"""
    component = Component.query.get(component_id)
    if not component:
        return None
    
    for key, value in kwargs.items():
        if key == 'hylleplass' and value:
            value = value.upper()  # ENFORCE CAPS
        if hasattr(component, key):
            setattr(component, key, value)
    
    db.session.commit()
    return component


def move_component(component_id: int, new_hylleplass: str):
    """Move component to new location"""
    return update_component(component_id, hylleplass=new_hylleplass.upper())


def delete_component(component_id: int):
    """Delete a component"""
    component = Component.query.get(component_id)
    if component:
        db.session.delete(component)
        db.session.commit()
        return True
    return False


def get_all_hylleplasser():
    """Get list of all unique hylleplass values (for dropdowns)"""
    results = db.session.query(Component.hylleplass).distinct().order_by(Component.hylleplass).all()
    return [r[0] for r in results]
