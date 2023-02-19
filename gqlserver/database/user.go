package database

import (
	"time"
)

type User struct {
	ID uint `gorm:"primaryKey"`

	CreatedAt time.Time `gorm:"not null"`
	UpdatedAt time.Time `gorm:"not null"`

	Username     string `gorm:"not null;uniqueIndex"`
	PasswordHash string `gorm:"not null"`
	IsAdmin      bool   `gorm:"not null"`
}
