require("dotenv").config();
const mongoose = require("mongoose");
const User = require("./models/User");

// Team member admin accounts (loaded from environment variables for security)
const TEAM_ADMINS = [
    {
        name: process.env.ADMIN1_NAME || "Member 1",
        email: process.env.ADMIN1_EMAIL,
        password: process.env.ADMIN1_PASSWORD,
        role: "admin",
    },
    {
        name: process.env.ADMIN2_NAME || "Member 2",
        email: process.env.ADMIN2_EMAIL,
        password: process.env.ADMIN2_PASSWORD,
        role: "admin",
    },
    {
        name: process.env.ADMIN3_NAME || "Member 3",
        email: process.env.ADMIN3_EMAIL,
        password: process.env.ADMIN3_PASSWORD,
        role: "admin",
    },
];

async function createTeamAdmins() {
    try {
        await mongoose.connect(process.env.MONGODB_URI);
        console.log("MongoDB connected\n");

        let created = 0;
        let existing = 0;

        for (const adminData of TEAM_ADMINS) {
            if (!adminData.email || !adminData.password) {
                console.log(
                    `⚠️  Skipping admin (missing credentials in .env): ${adminData.name}`,
                );
                continue;
            }

            // Check if admin already exists
            const existingAdmin = await User.findOne({
                email: adminData.email,
            });

            if (existingAdmin) {
                console.log(`⚠️  Admin already exists: ${adminData.email}`);
                existing++;
                continue;
            }

            // Create admin user
            const admin = await User.create(adminData);
            console.log(`✅ Created admin: ${admin.name} (${admin.email})`);
            created++;
        }

        console.log("\n" + "=".repeat(50));
        console.log(
            `📊 Summary: ${created} created, ${existing} already existed`,
        );
        console.log("=".repeat(50));

        console.log("\n🌐 Login at: https://stratumai.vercel.app/login");
        console.log("🎛️  Admin Panel: https://stratumai.vercel.app/admin\n");

        process.exit(0);
    } catch (err) {
        console.error("❌ Error creating team admins:", err.message);
        process.exit(1);
    }
}

createTeamAdmins();
